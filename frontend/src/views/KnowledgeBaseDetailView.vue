<script setup lang="ts">
import { useMutation, useQuery, useQueryClient } from "@tanstack/vue-query";
import { computed, ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import {
  ElAlert,
  ElButton,
  ElCard,
  ElEmpty,
  ElMessage,
  ElMessageBox,
  ElPagination,
  ElSkeleton,
  ElTable,
  ElTableColumn,
  ElTag,
} from "element-plus";

import {
  deleteDocument,
  fetchDocuments,
  fetchKnowledgeBase,
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
  onSuccess: () =>
    queryClient.invalidateQueries({ queryKey: ["documents", knowledgeBaseId.value] }),
});

const deleteMutation = useMutation({
  mutationFn: (documentId: string) => deleteDocument(knowledgeBaseId.value, documentId),
  onSuccess: () =>
    queryClient.invalidateQueries({ queryKey: ["documents", knowledgeBaseId.value] }),
});

const isLoading = computed(() => knowledgeBaseQuery.isPending.value);
const isError = computed(() => knowledgeBaseQuery.isError.value);
const errorMessage = computed(() =>
  getErrorMessage(knowledgeBaseQuery.error.value, "无法加载知识库，请稍后重试。"),
);
const documents = computed(() => documentsQuery.data.value?.items ?? []);
const documentTotal = computed(() => documentsQuery.data.value?.total ?? 0);
const documentsLoading = computed(() => documentsQuery.isPending.value);
const documentsError = computed(() => documentsQuery.isError.value);
const documentsErrorMessage = computed(() =>
  getErrorMessage(documentsQuery.error.value, "无法加载文档，请稍后重试。"),
);
const isUploading = computed(() => uploadMutation.isPending.value);
const isDeletingDocument = computed(() => deleteMutation.isPending.value);
const isDocumentActionPending = computed(() => isUploading.value || isDeletingDocument.value);

function getErrorMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

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
    ElMessage.error(getErrorMessage(error, "文档上传失败，请稍后重试。"));
  }
}

async function confirmDeleteDocument(document: Document): Promise<void> {
  try {
    await ElMessageBox.confirm(
      `确定删除文档“${document.original_filename}”吗？删除后无法恢复。`,
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
    ElMessage.error(getErrorMessage(reason, "删除文档失败，请稍后重试。"));
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
    ElMessage.error(getErrorMessage(error, "文档删除失败，请稍后重试。"));
  } finally {
    deletingDocumentId.value = null;
  }
}

function changePage(page: number): void {
  currentPage.value = page;
}

function formatSize(sizeBytes: number): string {
  if (sizeBytes < 1024) {
    return `${sizeBytes} B`;
  }
  if (sizeBytes < 1024 * 1024) {
    return `${(sizeBytes / 1024).toFixed(1)} KB`;
  }
  return `${(sizeBytes / (1024 * 1024)).toFixed(1)} MB`;
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
    <div class="detail-toolbar">
      <el-button
        text
        @click="goBack"
      >
        ← 返回知识库
      </el-button>
    </div>

    <div
      v-if="isLoading"
      class="state-panel"
    >
      <el-skeleton
        :rows="7"
        animated
      />
    </div>

    <div
      v-else-if="isError"
      class="state-panel error-state"
    >
      <el-alert
        title="暂时无法加载知识库"
        :description="errorMessage"
        type="error"
        :closable="false"
        show-icon
      />
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
      <div class="page-heading">
        <div>
          <p class="eyebrow">
            知识库
          </p>
          <h1>{{ knowledgeBase.name }}</h1>
          <p class="page-description">
            查看知识库基本信息和已上传的 PDF 文档。
          </p>
        </div>
      </div>

      <el-card
        class="detail-card"
        shadow="never"
      >
        <div class="description-block">
          <span class="field-label">描述</span>
          <p>{{ knowledgeBase.description || "未填写描述" }}</p>
        </div>
        <div class="metadata-grid">
          <div class="metadata-item">
            <span class="field-label">创建时间</span>
            <strong>{{ formatDate(knowledgeBase.created_at) }}</strong>
          </div>
          <div class="metadata-item">
            <span class="field-label">更新时间</span>
            <strong>{{ formatDate(knowledgeBase.updated_at) }}</strong>
          </div>
        </div>
      </el-card>

      <el-card
        class="documents-card"
        shadow="never"
      >
        <div class="documents-toolbar">
          <div>
            <h2>文档</h2>
            <p>当前知识库中的 PDF 文件</p>
          </div>
          <div class="upload-action">
            <input
              ref="fileInput"
              class="visually-hidden"
              type="file"
              accept=".pdf,application/pdf"
              @change="onFileSelected"
            >
            <el-button
              type="primary"
              :loading="isUploading"
              :disabled="isDocumentActionPending"
              @click="chooseDocument"
            >
              上传 PDF
            </el-button>
          </div>
        </div>

        <p
          v-if="isUploading"
          class="pending-hint"
        >
          正在上传文档，请稍候…
        </p>

        <div
          v-if="documentsLoading"
          class="documents-state"
        >
          <el-skeleton
            :rows="4"
            animated
          />
        </div>

        <div
          v-else-if="documentsError"
          class="documents-state error-state"
        >
          <el-alert
            title="暂时无法加载文档"
            :description="documentsErrorMessage"
            type="error"
            :closable="false"
            show-icon
          />
          <el-button @click="retryDocuments">
            重试
          </el-button>
        </div>

        <div
          v-else-if="documents.length === 0"
          class="documents-state empty-state"
        >
          <el-empty description="还没有文档">
            <el-button
              type="primary"
              :disabled="isDocumentActionPending"
              @click="chooseDocument"
            >
              上传第一个 PDF
            </el-button>
          </el-empty>
        </div>

        <template v-else>
          <el-table
            :data="documents"
            row-key="id"
            class="documents-table"
          >
            <el-table-column
              label="文件名"
              min-width="260"
            >
              <template #default="scope">
                <span class="document-name">{{ scope.row.original_filename }}</span>
              </template>
            </el-table-column>
            <el-table-column
              label="大小"
              width="120"
            >
              <template #default="scope">
                <span class="document-meta">{{ formatSize(scope.row.size_bytes) }}</span>
              </template>
            </el-table-column>
            <el-table-column
              label="状态"
              width="120"
            >
              <template #default="scope">
                <el-tag
                  type="success"
                  effect="plain"
                >
                  {{ statusLabel(scope.row.status) }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column
              label="上传时间"
              width="190"
            >
              <template #default="scope">
                <span class="document-meta">{{ formatDate(scope.row.created_at) }}</span>
              </template>
            </el-table-column>
            <el-table-column
              label="操作"
              width="88"
              align="right"
            >
              <template #default="scope">
                <el-button
                  link
                  type="danger"
                  :loading="isDeletingDocument && deletingDocumentId === scope.row.id"
                  :disabled="isDocumentActionPending"
                  @click="confirmDeleteDocument(toDocument(scope.row))"
                >
                  删除
                </el-button>
              </template>
            </el-table-column>
          </el-table>

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
      </el-card>
    </template>
  </section>
</template>

<style scoped>
.knowledge-base-detail-page {
  display: flex;
  flex-direction: column;
  gap: 22px;
}

.detail-toolbar {
  min-height: 32px;
}

.page-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 24px;
}

.eyebrow {
  margin: 0 0 8px;
  color: #8290a0;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.14em;
}

h1,
h2,
p {
  margin-top: 0;
}

h1 {
  margin-bottom: 10px;
  color: #1e2c3d;
  font-size: clamp(28px, 3vw, 38px);
  font-weight: 700;
  letter-spacing: -0.03em;
}

h2 {
  margin-bottom: 5px;
  color: #2b3c50;
  font-size: 17px;
  font-weight: 650;
}

.page-description {
  margin-bottom: 0;
  color: #7c8999;
  font-size: 14px;
}

.state-panel,
.documents-state {
  display: flex;
  min-height: 300px;
  flex-direction: column;
  justify-content: center;
  gap: 18px;
  padding: 28px;
  background: #ffffff;
  border: 1px solid #e7ebf0;
  border-radius: 12px;
}

.documents-state {
  min-height: 230px;
  padding: 28px 8px;
}

.error-state {
  align-items: flex-start;
}

.empty-state {
  align-items: center;
}

.error-actions {
  display: flex;
  gap: 10px;
}

.detail-card,
.documents-card {
  border-color: #e7ebf0;
  border-radius: 12px;
}

.description-block {
  padding-bottom: 28px;
  border-bottom: 1px solid #edf0f3;
}

.description-block p {
  margin: 10px 0 0;
  color: #4b5d71;
  font-size: 15px;
  line-height: 1.8;
  white-space: pre-wrap;
}

.field-label {
  color: #8491a1;
  font-size: 12px;
}

.metadata-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 28px;
  padding-top: 28px;
}

.metadata-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.metadata-item strong {
  color: #34485e;
  font-size: 14px;
  font-weight: 600;
}

.documents-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  padding-bottom: 20px;
  border-bottom: 1px solid #edf0f3;
}

.documents-toolbar p {
  margin-bottom: 0;
  color: #8a96a6;
  font-size: 12px;
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
  margin: 16px 0 0;
  color: #71849a;
  font-size: 13px;
}

.documents-table {
  margin-top: 4px;
}

.document-name {
  color: #34485e;
  font-size: 14px;
  font-weight: 600;
  overflow-wrap: anywhere;
}

.document-meta {
  color: #718197;
  font-size: 13px;
}

.pagination-row {
  display: flex;
  justify-content: flex-end;
  padding-top: 22px;
}

@media (max-width: 560px) {
  .metadata-grid {
    grid-template-columns: 1fr;
    gap: 20px;
  }

  .documents-toolbar {
    align-items: flex-start;
    flex-direction: column;
  }
}
</style>
