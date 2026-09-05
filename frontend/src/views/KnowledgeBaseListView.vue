<script setup lang="ts">
import { useMutation, useQuery, useQueryClient } from "@tanstack/vue-query";
import { computed, reactive, ref } from "vue";
import { RouterLink } from "vue-router";
import {
  ElAlert,
  ElButton,
  ElCard,
  ElDialog,
  ElEmpty,
  ElForm,
  ElFormItem,
  ElInput,
  ElMessage,
  ElMessageBox,
  ElPagination,
  ElSkeleton,
  ElTable,
  ElTableColumn,
} from "element-plus";
import type { FormInstance, FormRules } from "element-plus";

import {
  createKnowledgeBase,
  deleteKnowledgeBase,
  fetchKnowledgeBases,
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
const errorMessage = computed(() => getErrorMessage(listQuery.error.value));

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

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "无法加载知识库，请稍后重试。";
}

function toKnowledgeBase(row: unknown): KnowledgeBase {
  return row as KnowledgeBase;
}

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
  if (!valid) {
    return;
  }
  if (isMutating.value) {
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
    ElMessage.error(getErrorMessage(error));
  }
}

async function confirmDelete(knowledgeBase: KnowledgeBase): Promise<void> {
  try {
    await ElMessageBox.confirm(
      `确定删除知识库“${knowledgeBase.name}”吗？删除后无法恢复。`,
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
    ElMessage.error(getErrorMessage(reason));
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
    ElMessage.error(getErrorMessage(error));
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
    <div class="page-heading">
      <div>
        <p class="eyebrow">
          知识工作区
        </p>
        <h1>知识库</h1>
        <p class="page-description">
          集中管理当前工作区中的知识库，为后续内容工作提供清晰边界。
        </p>
      </div>
      <el-button
        type="primary"
        :disabled="isMutating"
        @click="openCreateDialog"
      >
        新建知识库
      </el-button>
    </div>

    <el-card
      class="list-card"
      shadow="never"
    >
      <div class="list-toolbar">
        <div>
          <h2>全部知识库</h2>
          <p>共 {{ total }} 个知识库</p>
        </div>
      </div>

      <div
        v-if="isLoading"
        class="state-panel"
      >
        <el-skeleton
          :rows="5"
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
        <el-button @click="retry">
          重试
        </el-button>
      </div>

      <div
        v-else-if="knowledgeBases.length === 0"
        class="state-panel empty-state"
      >
        <el-empty description="还没有知识库">
          <el-button
            type="primary"
            @click="openCreateDialog"
          >
            创建第一个知识库
          </el-button>
        </el-empty>
      </div>

      <template v-else>
        <el-table
          :data="knowledgeBases"
          row-key="id"
          class="knowledge-base-table"
        >
          <el-table-column
            label="名称"
            min-width="260"
          >
            <template #default="scope">
              <RouterLink
                :to="{ name: 'knowledge-base-detail', params: { id: scope.row.id } }"
                class="knowledge-base-name"
              >
                {{ scope.row.name }}
              </RouterLink>
            </template>
          </el-table-column>
          <el-table-column
            label="描述"
            min-width="320"
          >
            <template #default="scope">
              <span class="description-cell">{{ scope.row.description || "未填写描述" }}</span>
            </template>
          </el-table-column>
          <el-table-column
            label="更新时间"
            width="190"
          >
            <template #default="scope">
              <span class="date-cell">{{ formatDate(scope.row.updated_at) }}</span>
            </template>
          </el-table-column>
          <el-table-column
            label="操作"
            width="150"
            align="right"
          >
            <template #default="scope">
              <div class="row-actions">
                <el-button
                  link
                  type="primary"
                  :disabled="isMutating"
                  @click="openEditDialog(toKnowledgeBase(scope.row))"
                >
                  编辑
                </el-button>
                <el-button
                  link
                  type="danger"
                  :disabled="isMutating"
                  @click="confirmDelete(toKnowledgeBase(scope.row))"
                >
                  删除
                </el-button>
              </div>
            </template>
          </el-table-column>
        </el-table>

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
    </el-card>

    <el-dialog
      v-model="dialogVisible"
      :title="dialogTitle"
      width="520px"
      :close-on-click-modal="false"
    >
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
  gap: 24px;
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

.page-description {
  max-width: 680px;
  margin-bottom: 0;
  color: #7c8999;
  font-size: 14px;
  line-height: 1.7;
}

.list-card {
  border-color: #e7ebf0;
  border-radius: 12px;
}

.list-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding-bottom: 20px;
  border-bottom: 1px solid #edf0f3;
}

.list-toolbar h2 {
  margin-bottom: 5px;
  color: #2b3c50;
  font-size: 17px;
  font-weight: 650;
}

.list-toolbar p {
  margin-bottom: 0;
  color: #8a96a6;
  font-size: 12px;
}

.state-panel {
  display: flex;
  min-height: 300px;
  flex-direction: column;
  justify-content: center;
  gap: 18px;
  padding: 28px 8px;
}

.error-state {
  align-items: flex-start;
}

.empty-state {
  align-items: center;
}

.knowledge-base-table {
  margin-top: 4px;
}

.knowledge-base-name {
  color: #294c70;
  font-size: 14px;
  font-weight: 650;
}

.knowledge-base-name:hover {
  color: #1a3653;
  text-decoration: underline;
  text-underline-offset: 3px;
}

.description-cell,
.date-cell {
  color: #748397;
  font-size: 13px;
}

.description-cell {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.row-actions {
  display: inline-flex;
  gap: 4px;
}

.pagination-row {
  display: flex;
  justify-content: flex-end;
  padding-top: 22px;
}

@media (max-width: 680px) {
  .page-heading {
    align-items: flex-start;
    flex-direction: column;
  }

  .list-card :deep(.el-card__body) {
    padding: 16px;
  }

  .pagination-row {
    justify-content: center;
  }
}
</style>
