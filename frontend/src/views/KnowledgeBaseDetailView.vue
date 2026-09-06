<script setup lang="ts">
import { useMutation, useQuery, useQueryClient } from "@tanstack/vue-query";
import { computed, ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import {
  ElButton,
  ElDropdown,
  ElDropdownItem,
  ElDropdownMenu,
  ElMessage,
  ElMessageBox,
  ElPagination,
  ElTag,
} from "element-plus";

import {
  deleteDocument,
  fetchDocuments,
  fetchKnowledgeBase,
  getUserFacingError,
  uploadDocument,
} from "../api/client";
import type { Document } from "../api/types";

const DOCUMENT_PAGE_SIZE = 10;

const route = useRoute();
const router = useRouter();
const queryClient = useQueryClient();
const knowledgeBaseId = computed(() => String(route.params.id));
const currentPage = ref(1);
const fileInput = ref<HTMLInputElement | null>(null);
const deletingDocumentId = ref<string | null>(null);

const knowledgeBaseQuery = useQuery({
  queryKey: computed(() => ["knowledge-base", knowledgeBaseId.value]),
  queryFn: () => fetchKnowledgeBase(knowledgeBaseId.value),
});

const knowledgeBase = computed(() => knowledgeBaseQuery.data.value);
const documentsQuery = useQuery({
  queryKey: computed(() => ["documents", knowledgeBaseId.value, currentPage.value]),
  queryFn: () =>
    fetchDocuments(knowledgeBaseId.value, {
      limit: DOCUMENT_PAGE_SIZE,
      offset: (currentPage.value - 1) * DOCUMENT_PAGE_SIZE,
    }),
  enabled: computed(() => Boolean(knowledgeBase.value)),
});

const uploadMutation = useMutation({
  mutationFn: (file: File) => uploadDocument(knowledgeBaseId.value, file),
  onSuccess: () => {
    currentPage.value = 1;
    return queryClient.invalidateQueries({ queryKey: ["documents", knowledgeBaseId.value] });
  },
});

const deleteMutation = useMutation({
  mutationFn: (documentId: string) => deleteDocument(knowledgeBaseId.value, documentId),
  onSuccess: () =>
    queryClient.invalidateQueries({ queryKey: ["documents", knowledgeBaseId.value] }),
});

const isLoading = computed(() => knowledgeBaseQuery.isPending.value);
const isError = computed(() => knowledgeBaseQuery.isError.value);
const errorMessage = computed(() =>
  getUserFacingError(knowledgeBaseQuery.error.value, "知识库暂时无法加载，请稍后重试。"),
);
const documents = computed(() => documentsQuery.data.value?.items ?? []);
const documentTotal = computed(() => documentsQuery.data.value?.total ?? 0);
const documentsLoading = computed(() => documentsQuery.isPending.value);
const documentsError = computed(() => documentsQuery.isError.value);
const documentsErrorMessage = computed(() =>
  getUserFacingError(documentsQuery.error.value, "文档暂时无法加载，请稍后重试。"),
);
const isUploading = computed(() => uploadMutation.isPending.value);
const isDeletingDocument = computed(() => deleteMutation.isPending.value);
const isDocumentActionPending = computed(() => isUploading.value || isDeletingDocument.value);

function toDocument(row: unknown): Document {
  return row as Document;
}

function goBack(): void {
  void router.push({ name: "knowledge-bases" });
}

async function retryKnowledgeBase(): Promise<void> {
  await knowledgeBaseQuery.refetch();
}

async function retryDocuments(): Promise<void> {
  await documentsQuery.refetch();
}

function chooseDocument(): void {
  if (!isDocumentActionPending.value) {
    fileInput.value?.click();
  }
}

async function onFileSelected(event: Event): Promise<void> {
  const input = event.currentTarget as HTMLInputElement;
  const file = input.files?.[0];
  input.value = "";
  if (!file) {
    return;
  }
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    ElMessage.error("仅支持 PDF 文件");
    return;
  }

  try {
    await uploadMutation.mutateAsync(file);
    ElMessage.success("文档已上传");
  } catch (error) {
    ElMessage.error(getUserFacingError(error, "文档上传失败，请稍后重试。"));
  }
}

function handleDocumentAction(command: string | number | object, document: Document): void {
  if (command === "delete") {
    void confirmDeleteDocument(document);
  }
}

async function confirmDeleteDocument(document: Document): Promise<void> {
  try {
    await ElMessageBox.confirm(
      "确定删除文档“" + document.original_filename + "”吗？删除后无法恢复。",
      "删除文档",
      {
        confirmButtonText: "删除",
        cancelButtonText: "取消",
        type: "warning",
      },
    );
  } catch (reason) {
    if (reason === "cancel" || reason === "close") {
      return;
    }
    ElMessage.error(getUserFacingError(reason, "删除文档失败，请稍后重试。"));
    return;
  }

  if (isDocumentActionPending.value) {
    return;
  }

  deletingDocumentId.value = document.id;
  const shouldMoveToPreviousPage = currentPage.value > 1 && documents.value.length === 1;
  try {
    await deleteMutation.mutateAsync(document.id);
    if (shouldMoveToPreviousPage) {
      currentPage.value -= 1;
    }
    ElMessage.success("文档已删除");
  } catch (error) {
    ElMessage.error(getUserFacingError(error, "文档删除失败，请稍后重试。"));
  } finally {
    deletingDocumentId.value = null;
  }
}

function changePage(page: number): void {
  currentPage.value = page;
}

function formatSize(sizeBytes: number): string {
  if (sizeBytes < 1024) {
    return sizeBytes + " B";
  }
  if (sizeBytes < 1024 * 1024) {
    return (sizeBytes / 1024).toFixed(1) + " KB";
  }
  return (sizeBytes / (1024 * 1024)).toFixed(1) + " MB";
}

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat("zh-CN", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function statusLabel(documentStatus: Document["status"]): string {
  return documentStatus === "uploaded" ? "已上传" : documentStatus;
}
</script>

<template>
  <section class="knowledge-base-detail-page">
    <div
      v-if="isLoading"
      class="state-panel detail-loading"
      role="status"
      aria-busy="true"
    >
      <span class="sr-only">正在加载知识库</span>
      <div class="detail-skeleton">
        <span class="skeleton-line skeleton-line-back" />
        <span class="skeleton-line skeleton-line-title" />
        <span class="skeleton-line skeleton-line-detail" />
        <span class="skeleton-line skeleton-line-meta" />
      </div>
    </div>

    <div
      v-else-if="isError"
      class="state-panel error-state"
      role="alert"
    >
      <div class="state-symbol state-symbol-error">
        !
      </div>
      <div class="state-copy">
        <h2>知识库暂时无法加载</h2>
        <p>{{ errorMessage }}</p>
      </div>
      <div class="error-actions">
        <el-button @click="goBack">
          返回知识库
        </el-button>
        <el-button
          type="primary"
          @click="retryKnowledgeBase"
        >
          重试
        </el-button>
      </div>
    </div>

    <template v-else-if="knowledgeBase">
      <header class="detail-header">
        <button
          class="back-link"
          type="button"
          @click="goBack"
        >
          <span aria-hidden="true">←</span>
          知识库
        </button>
        <div class="detail-title-row">
          <span
            class="detail-mark"
            aria-hidden="true"
          >KB</span>
          <div class="detail-title-copy">
            <h1>{{ knowledgeBase.name }}</h1>
            <p class="detail-description">
              {{ knowledgeBase.description || "尚未添加描述" }}
            </p>
          </div>
        </div>
        <div class="detail-meta">
          <span>更新于 {{ formatDate(knowledgeBase.updated_at) }}</span>
        </div>
      </header>

      <section
        class="documents-surface"
        aria-labelledby="documents-title"
      >
        <div class="documents-toolbar">
          <div>
            <div class="section-title-row">
              <h2 id="documents-title">
                文档
              </h2>
              <span class="document-count">{{ documentTotal }} 个文件</span>
            </div>
            <p>该知识库中的 PDF 文件</p>
          </div>
          <div class="upload-action">
            <input
              ref="fileInput"
              class="visually-hidden"
              type="file"
              accept=".pdf,application/pdf"
              aria-label="选择要上传的 PDF 文件"
              @change="onFileSelected"
            >
            <el-button
              type="primary"
              :loading="isUploading"
              :disabled="isDocumentActionPending"
              @click="chooseDocument"
            >
              <span
                class="button-leading"
                aria-hidden="true"
              >+</span>
              上传 PDF
            </el-button>
          </div>
        </div>

        <p
          v-if="isUploading"
          class="pending-hint"
          role="status"
        >
          <span
            class="pending-dot"
            aria-hidden="true"
          />
          正在上传文档…
        </p>

        <div
          v-if="documentsLoading"
          class="documents-state documents-loading"
          role="status"
          aria-busy="true"
        >
          <span class="sr-only">正在加载文档</span>
          <div class="document-skeleton-list">
            <div
              v-for="index in 3"
              :key="index"
              class="document-skeleton-row"
            >
              <span class="skeleton-mark skeleton-mark-pdf" />
              <span class="skeleton-copy">
                <span class="skeleton-line skeleton-line-document-name" />
                <span class="skeleton-line skeleton-line-document-meta" />
              </span>
              <span class="skeleton-line skeleton-line-document-size" />
            </div>
          </div>
        </div>

        <div
          v-else-if="documentsError"
          class="documents-state error-state"
          role="alert"
        >
          <div class="state-symbol state-symbol-error">
            !
          </div>
          <div class="state-copy">
            <h3>文档暂时无法加载</h3>
            <p>{{ documentsErrorMessage }}</p>
          </div>
          <el-button @click="retryDocuments">
            重试
          </el-button>
        </div>

        <div
          v-else-if="documents.length === 0"
          class="documents-state empty-state"
        >
          <div class="state-symbol state-symbol-empty">
            PDF
          </div>
          <div class="state-copy">
            <h3>还没有文档</h3>
            <p>上传一个 PDF，开始整理这个知识库。</p>
          </div>
          <el-button
            type="primary"
            :disabled="isDocumentActionPending"
            @click="chooseDocument"
          >
            上传第一个 PDF
          </el-button>
        </div>

        <template v-else>
          <div
            class="document-list"
            role="list"
          >
            <article
              v-for="document in documents"
              :key="document.id"
              class="document-row"
              role="listitem"
            >
              <div class="document-primary">
                <span
                  class="pdf-mark"
                  aria-hidden="true"
                >PDF</span>
                <div class="document-title">
                  <strong>{{ document.original_filename }}</strong>
                  <span>PDF 文档</span>
                </div>
              </div>
              <div class="document-details">
                <span>{{ formatSize(document.size_bytes) }}</span>
                <el-tag
                  type="success"
                  effect="plain"
                >
                  {{ statusLabel(document.status) }}
                </el-tag>
                <time :datetime="document.created_at">{{ formatDate(document.created_at) }}</time>
              </div>
              <el-dropdown
                trigger="click"
                placement="bottom-end"
                @command="handleDocumentAction($event, toDocument(document))"
              >
                <button
                  class="more-button"
                  type="button"
                  :aria-label="document.original_filename + ' 的更多操作'"
                  :disabled="isDocumentActionPending"
                  @click.stop
                >
                  <span
                    v-if="isDeletingDocument && deletingDocumentId === document.id"
                    class="mini-spinner"
                    aria-label="正在删除"
                  />
                  <span
                    v-else
                    aria-hidden="true"
                  >•••</span>
                </button>
                <template #dropdown>
                  <el-dropdown-menu>
                    <el-dropdown-item
                      command="delete"
                      :disabled="isDocumentActionPending"
                    >
                      删除
                    </el-dropdown-item>
                  </el-dropdown-menu>
                </template>
              </el-dropdown>
            </article>
          </div>

          <div
            v-if="documentTotal > DOCUMENT_PAGE_SIZE"
            class="pagination-row"
          >
            <el-pagination
              :current-page="currentPage"
              :page-size="DOCUMENT_PAGE_SIZE"
              :total="documentTotal"
              background
              layout="prev, pager, next"
              @current-change="changePage"
            />
          </div>
        </template>
      </section>
    </template>
  </section>
</template>

<style scoped>
.knowledge-base-detail-page {
  display: flex;
  flex-direction: column;
  gap: 28px;
}

h1,
h2,
h3,
p {
  margin-top: 0;
}

.detail-header {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.back-link {
  display: inline-flex;
  align-items: center;
  align-self: flex-start;
  gap: 7px;
  padding: 4px 0;
  color: var(--ks-muted);
  font-size: 13px;
  background: transparent;
  border: 0;
  cursor: pointer;
  transition: color var(--ks-duration-fast) var(--ks-ease-out);
}

.back-link:hover {
  color: var(--ks-accent-strong);
}

.back-link span {
  font-size: 17px;
  line-height: 1;
}

.detail-title-row {
  display: flex;
  align-items: flex-start;
  gap: 16px;
}

.detail-mark {
  display: grid;
  width: 48px;
  height: 48px;
  flex: 0 0 48px;
  place-items: center;
  color: var(--ks-accent-strong);
  font-size: 12px;
  font-weight: 720;
  letter-spacing: 0.03em;
  background: var(--ks-accent-soft);
  border-radius: 11px;
}

.detail-title-copy {
  min-width: 0;
}

h1 {
  margin-bottom: 8px;
  overflow-wrap: anywhere;
  color: var(--ks-ink);
  font-size: clamp(26px, 3vw, 32px);
  font-weight: 700;
  letter-spacing: -0.03em;
  line-height: 1.2;
}

.detail-description {
  max-width: 760px;
  margin-bottom: 0;
  color: var(--ks-muted);
  font-size: 14px;
  line-height: 1.7;
  white-space: pre-wrap;
}

.detail-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 20px;
  padding-left: 64px;
  color: var(--ks-muted);
  font-size: 12px;
}

.documents-surface {
  overflow: hidden;
  background: var(--ks-surface);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-lg);
  box-shadow: var(--ks-shadow-sm);
}

.documents-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  min-height: 82px;
  padding: 20px 24px;
  border-bottom: 1px solid var(--ks-border);
}

.section-title-row {
  display: flex;
  align-items: center;
  gap: 10px;
}

.section-title-row h2 {
  margin-bottom: 6px;
  color: var(--ks-ink);
  font-size: 17px;
  font-weight: 680;
  line-height: 1.3;
}

.document-count {
  align-self: flex-start;
  padding: 3px 7px;
  color: var(--ks-muted);
  font-size: 11px;
  line-height: 1.2;
  background: var(--ks-surface-muted);
  border-radius: 999px;
}

.documents-toolbar p {
  margin-bottom: 0;
  color: var(--ks-muted);
  font-size: 12px;
}

.button-leading {
  margin-right: 6px;
  font-size: 17px;
  font-weight: 400;
  line-height: 1;
}

.visually-hidden {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0 0 0 0);
  white-space: nowrap;
  clip-path: inset(50%);
}

.pending-hint {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 0;
  padding: 12px 24px;
  color: var(--ks-accent-strong);
  font-size: 13px;
  background: var(--ks-accent-soft);
  border-bottom: 1px solid var(--ks-border);
}

.pending-dot {
  width: 7px;
  height: 7px;
  background: var(--ks-accent);
  border-radius: 50%;
  animation: pending-pulse 1.1s ease-in-out infinite;
}

.document-list {
  padding: 0 24px;
}

.document-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto 34px;
  align-items: center;
  gap: 20px;
  min-height: 82px;
  border-bottom: 1px solid var(--ks-border);
}

.document-row:last-child {
  border-bottom: 0;
}

.document-primary {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 14px;
}

.pdf-mark {
  display: grid;
  width: 38px;
  height: 38px;
  flex: 0 0 38px;
  place-items: center;
  color: #a7524c;
  font-size: 10px;
  font-weight: 740;
  letter-spacing: 0.03em;
  background: #f8eae8;
  border-radius: 8px;
}

.document-title {
  display: flex;
  min-width: 0;
  flex-direction: column;
  gap: 5px;
}

.document-title strong {
  overflow: hidden;
  color: var(--ks-ink);
  font-size: 14px;
  font-weight: 650;
  line-height: 1.4;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.document-title span {
  color: var(--ks-muted);
  font-size: 12px;
}

.document-details {
  display: flex;
  align-items: center;
  gap: 22px;
  color: var(--ks-muted);
  font-size: 12px;
  white-space: nowrap;
}

.document-details .el-tag {
  --el-tag-bg-color: var(--ks-accent-soft);
  --el-tag-border-color: #c5dbd3;
  --el-tag-text-color: var(--ks-success);
  font-size: 11px;
}

.more-button {
  display: grid;
  width: 34px;
  height: 34px;
  place-items: center;
  padding: 0;
  color: var(--ks-muted);
  font-size: 16px;
  letter-spacing: 0.08em;
  background: transparent;
  border: 1px solid transparent;
  border-radius: var(--ks-radius-sm);
  cursor: pointer;
  transition: color var(--ks-duration-fast) var(--ks-ease-out),
    background-color var(--ks-duration-fast) var(--ks-ease-out),
    transform var(--ks-duration-fast) var(--ks-ease-out);
}

.more-button:hover:not(:disabled) {
  color: var(--ks-ink);
  background: var(--ks-surface-muted);
}

.more-button:active:not(:disabled) {
  transform: scale(0.94);
}

.more-button:disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

.mini-spinner {
  width: 14px;
  height: 14px;
  border: 2px solid var(--ks-border-strong);
  border-top-color: var(--ks-accent);
  border-radius: 50%;
  animation: spinner-rotate 800ms linear infinite;
}

.state-panel,
.documents-state {
  display: flex;
  min-height: 280px;
  align-items: center;
  justify-content: center;
  gap: 16px;
  padding: 32px 24px;
}

.state-copy {
  min-width: 0;
}

.state-copy h2,
.state-copy h3 {
  margin-bottom: 6px;
  color: var(--ks-ink);
  font-size: 15px;
  font-weight: 680;
}

.state-copy p {
  margin-bottom: 0;
  color: var(--ks-muted);
  font-size: 13px;
  line-height: 1.6;
}

.state-symbol {
  display: grid;
  width: 44px;
  height: 44px;
  flex: 0 0 44px;
  place-items: center;
  color: var(--ks-accent-strong);
  font-size: 11px;
  font-weight: 720;
  letter-spacing: 0.03em;
  background: var(--ks-accent-soft);
  border-radius: 9px;
}

.state-symbol-error {
  color: var(--ks-danger);
  background: #f8eae8;
}

.error-state {
  justify-content: flex-start;
}

.error-actions {
  display: flex;
  gap: 8px;
}

.empty-state {
  flex-direction: column;
  text-align: center;
}

.empty-state .state-copy {
  display: flex;
  align-items: center;
  flex-direction: column;
}

.detail-loading {
  min-height: 460px;
}

.detail-skeleton {
  display: flex;
  width: 100%;
  max-width: 760px;
  flex-direction: column;
  gap: 16px;
}

.document-skeleton-list {
  width: 100%;
}

.document-skeleton-row {
  display: grid;
  grid-template-columns: 38px minmax(0, 1fr) 120px;
  align-items: center;
  gap: 14px;
  min-height: 82px;
  border-bottom: 1px solid var(--ks-border);
}

.skeleton-mark,
.skeleton-line {
  display: block;
  background: var(--ks-surface-muted);
  animation: skeleton-pulse 1.4s ease-in-out infinite;
}

.skeleton-line {
  height: 10px;
  border-radius: 4px;
}

.skeleton-line-back {
  width: 80px;
}

.skeleton-line-title {
  width: 42%;
  height: 24px;
}

.skeleton-line-detail {
  width: 64%;
}

.skeleton-line-meta {
  width: 32%;
}

.skeleton-mark-pdf {
  width: 38px;
  height: 38px;
  border-radius: 8px;
}

.skeleton-copy {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.skeleton-line-document-name {
  width: 56%;
}

.skeleton-line-document-meta {
  width: 26%;
}

.skeleton-line-document-size {
  width: 80px;
}

.pagination-row {
  display: flex;
  justify-content: flex-end;
  padding: 20px 24px 22px;
  border-top: 1px solid var(--ks-border);
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

@keyframes pending-pulse {
  0%,
  100% {
    opacity: 0.45;
  }

  50% {
    opacity: 1;
  }
}

@keyframes spinner-rotate {
  to {
    transform: rotate(360deg);
  }
}

@keyframes skeleton-pulse {
  0% {
    opacity: 0.55;
  }

  100% {
    opacity: 1;
  }
}

@media (max-width: 760px) {
  .documents-toolbar {
    align-items: flex-start;
    flex-direction: column;
    padding: 18px;
  }

  .document-list {
    padding: 0 18px;
  }

  .document-row {
    grid-template-columns: minmax(0, 1fr) 34px;
    gap: 10px 12px;
    padding: 14px 0;
  }

  .document-details {
    grid-column: 1 / -1;
    gap: 14px;
    padding-left: 52px;
  }

  .document-details time {
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .pagination-row {
    justify-content: center;
    padding: 18px;
  }

  .detail-meta {
    padding-left: 64px;
  }
}

@media (max-width: 560px) {
  .detail-title-row {
    gap: 12px;
  }

  .detail-mark {
    width: 42px;
    height: 42px;
    flex-basis: 42px;
  }

  .detail-meta {
    padding-left: 54px;
  }

  .state-panel,
  .documents-state {
    min-height: 250px;
    flex-direction: column;
    text-align: center;
  }

  .error-state {
    align-items: center;
  }

  .error-actions {
    justify-content: center;
  }

  .document-details {
    flex-wrap: wrap;
  }

  .document-skeleton-row {
    grid-template-columns: 38px minmax(0, 1fr);
  }

  .skeleton-line-document-size {
    display: none;
  }
}

@media (prefers-reduced-motion: reduce) {
  .pending-dot,
  .mini-spinner,
  .skeleton-mark,
  .skeleton-line {
    animation: none;
  }
}
</style>
