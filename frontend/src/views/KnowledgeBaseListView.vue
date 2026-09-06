<script setup lang="ts">
import { useMutation, useQuery, useQueryClient } from "@tanstack/vue-query";
import { computed, reactive, ref } from "vue";
import { RouterLink } from "vue-router";
import {
  ElButton,
  ElDialog,
  ElDropdown,
  ElDropdownItem,
  ElDropdownMenu,
  ElForm,
  ElFormItem,
  ElInput,
  ElMessage,
  ElMessageBox,
  ElPagination,
} from "element-plus";
import type { FormInstance, FormRules } from "element-plus";

import {
  createKnowledgeBase,
  deleteKnowledgeBase,
  fetchKnowledgeBases,
  getUserFacingError,
  updateKnowledgeBase,
} from "../api/client";
import type {
  KnowledgeBase,
  KnowledgeBaseCreateRequest,
  KnowledgeBaseUpdateRequest,
} from "../api/types";

const PAGE_SIZE = 10;

const queryClient = useQueryClient();
const currentPage = ref(1);
const listQuery = useQuery({
  queryKey: computed(() => ["knowledge-bases", currentPage.value, PAGE_SIZE]),
  queryFn: () =>
    fetchKnowledgeBases({
      limit: PAGE_SIZE,
      offset: (currentPage.value - 1) * PAGE_SIZE,
    }),
});

const knowledgeBases = computed(() => listQuery.data.value?.items ?? []);
const total = computed(() => listQuery.data.value?.total ?? 0);
const isLoading = computed(() => listQuery.isPending.value);
const isError = computed(() => listQuery.isError.value);
const errorMessage = computed(() =>
  getUserFacingError(listQuery.error.value, "知识库暂时无法加载，请稍后重试。"),
);

const dialogVisible = ref(false);
const dialogMode = ref<"create" | "edit">("create");
const editingId = ref<string | null>(null);
const formRef = ref<FormInstance>();
const form = reactive({
  name: "",
  description: "",
});

const formRules: FormRules = {
  name: [
    { required: true, whitespace: true, message: "请输入知识库名称", trigger: "blur" },
    { max: 200, message: "名称不能超过 200 个字符", trigger: "blur" },
  ],
  description: [{ max: 2_000, message: "描述不能超过 2000 个字符", trigger: "blur" }],
};

const dialogTitle = computed(() => (dialogMode.value === "create" ? "新建知识库" : "编辑知识库"));
const dialogDescription = computed(() =>
  dialogMode.value === "create" ? "为文档整理建立一个清晰的内容边界。" : "更新知识库名称或描述。",
);
const isMutating = computed(
  () =>
    createMutation.isPending.value ||
    updateMutation.isPending.value ||
    deleteMutation.isPending.value,
);

const createMutation = useMutation({
  mutationFn: (payload: KnowledgeBaseCreateRequest) => createKnowledgeBase(payload),
  onSuccess: () => queryClient.invalidateQueries({ queryKey: ["knowledge-bases"] }),
});

const updateMutation = useMutation({
  mutationFn: ({ id, payload }: { id: string; payload: KnowledgeBaseUpdateRequest }) =>
    updateKnowledgeBase(id, payload),
  onSuccess: (_data, variables) =>
    Promise.all([
      queryClient.invalidateQueries({ queryKey: ["knowledge-bases"] }),
      queryClient.invalidateQueries({ queryKey: ["knowledge-base", variables.id] }),
    ]),
});

const deleteMutation = useMutation({
  mutationFn: (id: string) => deleteKnowledgeBase(id),
  onSuccess: (_data, id) =>
    Promise.all([
      queryClient.invalidateQueries({ queryKey: ["knowledge-bases"] }),
      queryClient.invalidateQueries({ queryKey: ["knowledge-base", id] }),
    ]),
});

function resetForm(): void {
  form.name = "";
  form.description = "";
  editingId.value = null;
}

function openCreateDialog(): void {
  dialogMode.value = "create";
  resetForm();
  dialogVisible.value = true;
}

function openEditDialog(knowledgeBase: KnowledgeBase): void {
  dialogMode.value = "edit";
  editingId.value = knowledgeBase.id;
  form.name = knowledgeBase.name;
  form.description = knowledgeBase.description ?? "";
  dialogVisible.value = true;
}

function closeDialog(): void {
  if (!isMutating.value) {
    dialogVisible.value = false;
  }
}

async function submitForm(): Promise<void> {
  if (!formRef.value || isMutating.value) {
    return;
  }

  const valid = await formRef.value.validate().catch(() => false);
  if (!valid || isMutating.value) {
    return;
  }

  const payload = {
    name: form.name.trim(),
    description: form.description.trim() || null,
  };

  try {
    if (dialogMode.value === "create") {
      await createMutation.mutateAsync(payload);
      ElMessage.success("知识库已创建");
    } else if (editingId.value) {
      await updateMutation.mutateAsync({ id: editingId.value, payload });
      ElMessage.success("知识库已更新");
    }
    dialogVisible.value = false;
  } catch (error) {
    const fallback =
      dialogMode.value === "create" ? "知识库创建失败，请稍后重试。" : "知识库更新失败，请稍后重试。";
    ElMessage.error(getUserFacingError(error, fallback));
  }
}

function handleAction(command: string | number | object, knowledgeBase: KnowledgeBase): void {
  if (command === "edit") {
    openEditDialog(knowledgeBase);
  } else if (command === "delete") {
    void confirmDelete(knowledgeBase);
  }
}

async function confirmDelete(knowledgeBase: KnowledgeBase): Promise<void> {
  try {
    await ElMessageBox.confirm(
      "确定删除知识库“" + knowledgeBase.name + "”吗？删除后无法恢复。",
      "删除知识库",
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
    ElMessage.error(getUserFacingError(reason, "删除知识库失败，请稍后重试。"));
    return;
  }

  if (isMutating.value) {
    return;
  }

  const shouldMoveToPreviousPage = currentPage.value > 1 && knowledgeBases.value.length === 1;
  try {
    await deleteMutation.mutateAsync(knowledgeBase.id);
    if (shouldMoveToPreviousPage) {
      currentPage.value -= 1;
    }
    ElMessage.success("知识库已删除");
  } catch (error) {
    ElMessage.error(getUserFacingError(error, "删除知识库失败，请稍后重试。"));
  }
}

async function retry(): Promise<void> {
  await listQuery.refetch();
}

function changePage(page: number): void {
  currentPage.value = page;
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
</script>

<template>
  <section class="knowledge-base-page">
    <header class="page-heading">
      <div class="page-heading-copy">
        <h1>知识库</h1>
        <p class="page-description">
          管理行业文档，按主题整理你的知识内容。
        </p>
      </div>
      <el-button
        type="primary"
        :disabled="isMutating"
        @click="openCreateDialog"
      >
        <span
          class="button-leading"
          aria-hidden="true"
        >+</span>
        新建知识库
      </el-button>
    </header>

    <section class="collection-surface">
      <div
        v-if="isLoading"
        class="state-panel loading-state"
        role="status"
        aria-busy="true"
      >
        <span class="sr-only">正在加载知识库</span>
        <div class="kb-skeleton-list">
          <div
            v-for="index in 4"
            :key="index"
            class="kb-skeleton-row"
          >
            <span class="skeleton-mark" />
            <span class="skeleton-copy">
              <span class="skeleton-line skeleton-line-name" />
              <span class="skeleton-line skeleton-line-description" />
              <span class="skeleton-line skeleton-line-meta" />
            </span>
          </div>
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
          <h3>知识库暂时无法加载</h3>
          <p>{{ errorMessage }}</p>
        </div>
        <el-button @click="retry">
          重试
        </el-button>
      </div>

      <div
        v-else-if="knowledgeBases.length === 0"
        class="state-panel empty-state"
      >
        <div class="state-symbol state-symbol-empty">
          KB
        </div>
        <div class="state-copy">
          <h3>还没有知识库</h3>
          <p>创建一个知识库，开始整理你的行业文档。</p>
        </div>
        <el-button
          type="primary"
          @click="openCreateDialog"
        >
          新建知识库
        </el-button>
      </div>

      <template v-else>
        <div
          class="knowledge-base-list"
          role="list"
        >
          <article
            v-for="knowledgeBase in knowledgeBases"
            :key="knowledgeBase.id"
            class="knowledge-base-item"
            role="listitem"
          >
            <RouterLink
              :to="{ name: 'knowledge-base-detail', params: { id: knowledgeBase.id } }"
              class="knowledge-base-link"
              :aria-label="'打开知识库 ' + knowledgeBase.name"
            >
              <span
                class="knowledge-base-mark"
                aria-hidden="true"
              >KB</span>
              <span class="knowledge-base-copy">
                <strong class="knowledge-base-name">{{ knowledgeBase.name }}</strong>
                <span class="knowledge-base-description">
                  {{ knowledgeBase.description || "未填写描述" }}
                </span>
                <span class="knowledge-base-meta">
                  <span>最近更新</span>
                  <time :datetime="knowledgeBase.updated_at">{{ formatDate(knowledgeBase.updated_at) }}</time>
                </span>
              </span>
            </RouterLink>

            <el-dropdown
              trigger="click"
              placement="bottom-end"
              @command="handleAction($event, knowledgeBase)"
            >
              <button
                class="more-button"
                type="button"
                :aria-label="knowledgeBase.name + ' 的更多操作'"
                :disabled="isMutating"
                @click.stop
              >
                <span aria-hidden="true">•••</span>
              </button>
              <template #dropdown>
                <el-dropdown-menu>
                  <el-dropdown-item
                    command="edit"
                    :disabled="isMutating"
                  >
                    编辑
                  </el-dropdown-item>
                  <el-dropdown-item
                    command="delete"
                    :disabled="isMutating"
                  >
                    删除
                  </el-dropdown-item>
                </el-dropdown-menu>
              </template>
            </el-dropdown>
          </article>
        </div>

        <div
          v-if="total > PAGE_SIZE"
          class="pagination-row"
        >
          <el-pagination
            :current-page="currentPage"
            :page-size="PAGE_SIZE"
            :total="total"
            background
            layout="prev, pager, next"
            @current-change="changePage"
          />
        </div>
      </template>
    </section>

    <el-dialog
      v-model="dialogVisible"
      class="knowledge-base-dialog"
      :title="dialogTitle"
      width="480px"
      :close-on-click-modal="false"
    >
      <p class="dialog-description">
        {{ dialogDescription }}
      </p>
      <el-form
        ref="formRef"
        :model="form"
        :rules="formRules"
        label-position="top"
        @submit.prevent="submitForm"
      >
        <el-form-item
          label="名称"
          prop="name"
        >
          <el-input
            v-model="form.name"
            maxlength="200"
            show-word-limit
            placeholder="例如：质量管理规范"
          />
        </el-form-item>
        <el-form-item
          label="描述"
          prop="description"
        >
          <el-input
            v-model="form.description"
            type="textarea"
            :rows="4"
            maxlength="2000"
            show-word-limit
            placeholder="简要说明这个知识库的用途（可选）"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button
          :disabled="isMutating"
          @click="closeDialog"
        >
          取消
        </el-button>
        <el-button
          type="primary"
          :loading="isMutating"
          @click="submitForm"
        >
          保存
        </el-button>
      </template>
    </el-dialog>
  </section>
</template>

<style scoped>
.knowledge-base-page {
  display: flex;
  flex-direction: column;
  gap: 32px;
}

.page-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 24px;
}

h1,
h2,
h3,
p {
  margin-top: 0;
}

h1 {
  margin-bottom: 10px;
  color: var(--ks-ink);
  font-size: clamp(28px, 3vw, 32px);
  font-weight: 700;
  letter-spacing: -0.03em;
  line-height: 1.15;
}

.page-description {
  max-width: 680px;
  margin-bottom: 0;
  color: var(--ks-muted);
  font-size: 14px;
  line-height: 1.7;
}

.button-leading {
  margin-right: 6px;
  font-size: 17px;
  font-weight: 400;
  line-height: 1;
}

.collection-surface {
  overflow: hidden;
  background: var(--ks-surface);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-lg);
  box-shadow: var(--ks-shadow-sm);
}

.knowledge-base-list {
  padding: 0 24px;
}

.knowledge-base-item {
  display: flex;
  align-items: center;
  gap: 16px;
  min-height: 96px;
  border-bottom: 1px solid var(--ks-border);
}

.knowledge-base-item:last-child {
  border-bottom: 0;
}

.knowledge-base-link {
  display: flex;
  min-width: 0;
  flex: 1;
  align-items: center;
  gap: 16px;
  min-height: 72px;
  padding: 12px 8px 12px 0;
  border-radius: var(--ks-radius-sm);
  transition: background-color var(--ks-duration-fast) var(--ks-ease-out),
    transform var(--ks-duration-fast) var(--ks-ease-out);
}

.knowledge-base-link:hover {
  background: var(--ks-surface-subtle);
}

.knowledge-base-link:active {
  transform: scale(0.995);
}

.knowledge-base-mark,
.state-symbol {
  display: grid;
  flex: 0 0 auto;
  place-items: center;
  color: var(--ks-accent-strong);
  font-size: 11px;
  font-weight: 720;
  letter-spacing: 0.03em;
  background: var(--ks-accent-soft);
  border-radius: 9px;
}

.knowledge-base-mark {
  width: 40px;
  height: 40px;
}

.knowledge-base-copy {
  display: flex;
  min-width: 0;
  flex-direction: column;
  gap: 5px;
}

.knowledge-base-name {
  overflow: hidden;
  color: var(--ks-ink);
  font-size: 15px;
  font-weight: 680;
  line-height: 1.35;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.knowledge-base-link:hover .knowledge-base-name {
  color: var(--ks-accent-strong);
}

.knowledge-base-description {
  overflow: hidden;
  color: var(--ks-muted);
  font-size: 13px;
  line-height: 1.4;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.knowledge-base-meta {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--ks-faint);
  font-size: 12px;
  line-height: 1.3;
}

.knowledge-base-meta time {
  color: var(--ks-muted);
}

.more-button {
  display: grid;
  width: 34px;
  height: 34px;
  flex: 0 0 34px;
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

.state-panel {
  display: flex;
  min-height: 300px;
  align-items: center;
  justify-content: center;
  gap: 16px;
  padding: 32px 24px;
}

.state-copy {
  min-width: 0;
}

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
  width: 44px;
  height: 44px;
}

.state-symbol-error {
  color: var(--ks-danger);
  background: #f8eae8;
}

.error-state {
  justify-content: flex-start;
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

.loading-state {
  display: block;
  min-height: 380px;
}

.kb-skeleton-list {
  display: flex;
  flex-direction: column;
}

.kb-skeleton-row {
  display: flex;
  align-items: center;
  gap: 16px;
  min-height: 96px;
  border-bottom: 1px solid var(--ks-border);
}

.kb-skeleton-row:last-child {
  border-bottom: 0;
}

.skeleton-mark,
.skeleton-line {
  display: block;
  background: var(--ks-surface-muted);
  animation: skeleton-pulse 1.4s ease-in-out infinite;
}

.skeleton-mark {
  width: 40px;
  height: 40px;
  flex: 0 0 40px;
  border-radius: 9px;
}

.skeleton-copy {
  display: flex;
  flex: 1;
  flex-direction: column;
  gap: 8px;
}

.skeleton-line {
  height: 10px;
  border-radius: 4px;
}

.skeleton-line-name {
  width: 30%;
}

.skeleton-line-description {
  width: 58%;
}

.skeleton-line-meta {
  width: 20%;
}

.pagination-row {
  display: flex;
  justify-content: flex-end;
  padding: 20px 24px 22px;
  border-top: 1px solid var(--ks-border);
}

.dialog-description {
  margin-bottom: 22px;
  color: var(--ks-muted);
  font-size: 13px;
  line-height: 1.6;
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

@keyframes skeleton-pulse {
  0% {
    opacity: 0.55;
  }

  100% {
    opacity: 1;
  }
}

@media (max-width: 680px) {
  .page-heading {
    align-items: flex-start;
    flex-direction: column;
    gap: 18px;
  }

  .page-heading .el-button {
    align-self: flex-start;
  }

  .knowledge-base-list {
    padding: 0 18px;
  }

  .knowledge-base-item {
    gap: 8px;
  }

  .knowledge-base-link {
    gap: 12px;
  }

  .knowledge-base-description {
    max-width: 48vw;
  }

  .pagination-row {
    justify-content: center;
    padding: 18px;
  }

  .state-panel {
    flex-direction: column;
    min-height: 260px;
    text-align: center;
  }

  .error-state {
    align-items: center;
  }
}

@media (prefers-reduced-motion: reduce) {
  .skeleton-mark,
  .skeleton-line {
    animation: none;
  }
}
</style>
