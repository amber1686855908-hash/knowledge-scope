<script setup lang="ts">
import { useQuery } from "@tanstack/vue-query";
import { computed } from "vue";
import { useRoute, useRouter } from "vue-router";
import { ElAlert, ElButton, ElCard, ElSkeleton } from "element-plus";

import { fetchKnowledgeBase } from "../api/client";

const route = useRoute();
const router = useRouter();
const knowledgeBaseId = computed(() => String(route.params.id));
const knowledgeBaseQuery = useQuery({
  queryKey: computed(() => ["knowledge-base", knowledgeBaseId.value]),
  queryFn: () => fetchKnowledgeBase(knowledgeBaseId.value),
});

const knowledgeBase = computed(() => knowledgeBaseQuery.data.value);
const isLoading = computed(() => knowledgeBaseQuery.isPending.value);
const isError = computed(() => knowledgeBaseQuery.isError.value);
const errorMessage = computed(() => {
  const error = knowledgeBaseQuery.error.value;
  return error instanceof Error ? error.message : "无法加载知识库，请稍后重试。";
});

async function retry(): Promise<void> {
  await knowledgeBaseQuery.refetch();
}

function goBack(): void {
  void router.push({ name: "knowledge-bases" });
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
          @click="retry"
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
            知识库基本信息
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
  margin-bottom: 0;
  color: #7c8999;
  font-size: 14px;
}

.state-panel {
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

.error-state {
  align-items: flex-start;
}

.error-actions {
  display: flex;
  gap: 10px;
}

.detail-card {
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

@media (max-width: 560px) {
  .metadata-grid {
    grid-template-columns: 1fr;
    gap: 20px;
  }
}
</style>
