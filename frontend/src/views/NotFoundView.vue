<script setup lang="ts">
import { ElButton, ElResult } from "element-plus";
import { useRouter } from "vue-router";

const router = useRouter();

function goToDashboard(): void {
  void router.push("/");
}
</script>

<template>
  <div class="not-found-view">
    <el-result
      icon="warning"
      title="页面不存在"
      sub-title="当前版本仅提供 KnowledgeScope 项目基础概览。"
    >
      <template #extra>
        <el-button
          type="primary"
          @click="goToDashboard"
        >
          返回项目概览
        </el-button>
      </template>
    </el-result>
  </div>
</template>

<style scoped>
.not-found-view {
  display: grid;
  min-height: 440px;
  place-items: center;
  background: #ffffff;
  border: 1px solid #e7ebf0;
  border-radius: 12px;
}
</style>
