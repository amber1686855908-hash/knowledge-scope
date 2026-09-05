<script setup lang="ts">
import { useQuery } from "@tanstack/vue-query";
import { computed } from "vue";
import { ElAlert, ElButton, ElCard, ElSkeleton, ElTag } from "element-plus";

import { fetchHealth, fetchMeta } from "../api/client";
import type { HealthResponse, MetaResponse } from "../api/types";

const healthQuery = useQuery<HealthResponse>({
  queryKey: ["system", "health"],
  queryFn: fetchHealth,
});

const metaQuery = useQuery<MetaResponse>({
  queryKey: ["system", "meta"],
  queryFn: fetchMeta,
});

const health = computed(() => healthQuery.data.value);
const meta = computed(() => metaQuery.data.value);
const isLoading = computed(() => healthQuery.isPending.value || metaQuery.isPending.value);
const isError = computed(() => healthQuery.isError.value || metaQuery.isError.value);
const errorMessage = computed(() => {
  const error = healthQuery.error.value ?? metaQuery.error.value;
  return error instanceof Error ? error.message : "无法读取后端状态，请确认 API 服务正在运行。";
});

async function retry(): Promise<void> {
  await Promise.all([healthQuery.refetch(), metaQuery.refetch()]);
}
</script>

<template>
  <section class="dashboard-view">
    <div class="page-heading">
      <div>
        <p class="eyebrow">
          PROJECT FOUNDATION
        </p>
        <h1>项目概览</h1>
        <p class="page-description">
          这里展示 KnowledgeScope 当前已实现的 Web 应用基础状态，不包含尚未实现的业务数据。
        </p>
      </div>
      <el-tag
        type="info"
        effect="plain"
      >
        Phase A0.5
      </el-tag>
    </div>

    <div
      v-if="isLoading"
      class="state-panel"
    >
      <el-skeleton
        :rows="6"
        animated
      />
    </div>

    <div
      v-else-if="isError"
      class="state-panel error-state"
    >
      <el-alert
        title="暂时无法连接后端"
        :description="errorMessage"
        type="error"
        :closable="false"
        show-icon
      />
      <el-button
        type="primary"
        @click="retry"
      >
        重试
      </el-button>
    </div>

    <template v-else-if="health && meta">
      <div class="connection-card">
        <div>
          <p class="eyebrow">
            KNOWLEDGESCOPE
          </p>
          <h2>Web 应用基础已连接</h2>
          <p>前端已通过 `/api` 读取 FastAPI 的真实 health 和 meta 信息。</p>
        </div>
        <div class="connection-badge">
          <span
            class="connection-indicator"
            aria-hidden="true"
          >●</span>
          <span>Backend connected</span>
        </div>
      </div>

      <div class="info-grid">
        <el-card
          class="info-card"
          shadow="never"
        >
          <span class="card-label">项目版本</span>
          <strong class="card-value">{{ meta.version }}</strong>
          <span class="card-note">来自 /api/v1/meta</span>
        </el-card>
        <el-card
          class="info-card"
          shadow="never"
        >
          <span class="card-label">运行环境</span>
          <strong class="card-value">{{ health.environment }}</strong>
          <span class="card-note">Python {{ health.python_version }}</span>
        </el-card>
        <el-card
          class="info-card"
          shadow="never"
        >
          <span class="card-label">配置状态</span>
          <strong class="card-value">{{ health.config_status }}</strong>
          <span class="card-note">已通过后端配置校验</span>
        </el-card>
        <el-card
          class="info-card"
          shadow="never"
        >
          <span class="card-label">项目阶段</span>
          <strong class="card-value">{{ meta.phase }}</strong>
          <span class="card-note">{{ meta.status }}</span>
        </el-card>
      </div>

      <el-card
        class="detail-card"
        shadow="never"
      >
        <template #header>
          <div class="detail-heading">
            <span>运行信息</span>
            <span class="detail-caption">来自 /api/v1/health</span>
          </div>
        </template>
        <div class="detail-grid">
          <div class="detail-item">
            <span>项目名称</span>
            <strong>{{ health.project_name }}</strong>
          </div>
          <div class="detail-item">
            <span>日志级别</span>
            <strong>{{ health.log_level }}</strong>
          </div>
          <div class="detail-item">
            <span>数据目录</span>
            <strong>{{ health.data_dir }}</strong>
          </div>
          <div class="detail-item">
            <span>API 状态</span>
            <strong>{{ health.status }}</strong>
          </div>
        </div>
      </el-card>
    </template>
  </section>
</template>

<style scoped>
.dashboard-view {
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
  max-width: 660px;
  margin-bottom: 0;
  color: #7c8999;
  font-size: 14px;
  line-height: 1.7;
}

.state-panel {
  display: flex;
  min-height: 260px;
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

.connection-card {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  padding: 30px 34px;
  color: #ffffff;
  background: #24364b;
  border-radius: 14px;
  box-shadow: 0 12px 28px rgb(36 54 75 / 12%);
}

.connection-card .eyebrow {
  color: #9fb2c8;
}

.connection-card h2 {
  margin-bottom: 8px;
  font-size: 24px;
  font-weight: 650;
  letter-spacing: -0.02em;
}

.connection-card p:last-child {
  margin-bottom: 0;
  color: #c1cfdd;
  font-size: 13px;
}

.connection-badge {
  display: flex;
  flex: 0 0 auto;
  align-items: center;
  gap: 9px;
  padding: 10px 14px;
  color: #d8f0e3;
  font-size: 12px;
  font-weight: 650;
  background: rgb(255 255 255 / 10%);
  border: 1px solid rgb(255 255 255 / 13%);
  border-radius: 8px;
}

.connection-indicator {
  color: #71c293;
  font-size: 10px;
}

.info-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
}

.info-card {
  min-height: 138px;
  border-color: #e7ebf0;
  border-radius: 11px;
}

.info-card :deep(.el-card__body) {
  display: flex;
  min-height: 138px;
  flex-direction: column;
  align-items: flex-start;
  justify-content: center;
  gap: 8px;
}

.card-label,
.detail-item span {
  color: #8491a1;
  font-size: 12px;
}

.card-value {
  color: #24364b;
  font-size: 21px;
  font-weight: 700;
}

.card-note {
  overflow: hidden;
  max-width: 100%;
  color: #9aa5b2;
  font-size: 11px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.detail-card {
  border-color: #e7ebf0;
  border-radius: 11px;
}

.detail-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  color: #2b3c50;
  font-size: 14px;
  font-weight: 650;
}

.detail-caption {
  color: #9aa5b2;
  font-size: 11px;
  font-weight: 400;
}

.detail-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 24px;
}

.detail-item {
  display: flex;
  flex-direction: column;
  gap: 7px;
}

.detail-item strong {
  overflow: hidden;
  color: #3a4a5c;
  font-size: 13px;
  font-weight: 600;
  text-overflow: ellipsis;
  white-space: nowrap;
}

@media (max-width: 900px) {
  .info-grid,
  .detail-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .connection-card {
    align-items: flex-start;
    flex-direction: column;
  }
}

@media (max-width: 560px) {
  .page-heading {
    flex-direction: column;
  }

  .info-grid,
  .detail-grid {
    grid-template-columns: 1fr;
  }
}
</style>
