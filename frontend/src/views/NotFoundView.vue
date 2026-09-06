<script setup lang="ts">
import { ElButton, ElResult } from "element-plus";
import { useRouter } from "vue-router";

const router = useRouter();

function goToKnowledgeBases(): void {
  void router.push({ name: "knowledge-bases" });
}
</script>

<template>
  <div class="not-found-view">
    <el-result
      icon="warning"
      title="页面不存在"
      sub-title="找不到你要访问的内容。"
    >
      <template #extra>
        <el-button
          type="primary"
          @click="goToKnowledgeBases"
        >
          返回知识库
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
