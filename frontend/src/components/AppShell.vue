<script setup lang="ts">
import { computed } from "vue";
import { RouterLink } from "vue-router";

import { useUiStore } from "../stores/ui";

const uiStore = useUiStore();
const sidebarWidth = computed(() => (uiStore.sidebarCollapsed ? "76px" : "240px"));
</script>

<template>
  <div class="app-shell">
    <aside
      class="app-sidebar"
      :style="{ width: sidebarWidth }"
    >
      <div class="brand-block">
        <div
          class="brand-mark"
          aria-hidden="true"
        >
          KS
        </div>
        <div
          v-if="!uiStore.sidebarCollapsed"
          class="brand-copy"
        >
          <span class="brand-name">KnowledgeScope</span>
          <span class="brand-caption">Web foundation</span>
        </div>
      </div>

      <nav
        class="side-nav"
        aria-label="主导航"
      >
        <RouterLink
          to="/"
          class="nav-item"
          exact-active-class="is-active"
          title="项目概览"
        >
          <span
            class="nav-icon"
            aria-hidden="true"
          >⌂</span>
          <span v-if="!uiStore.sidebarCollapsed">项目概览</span>
        </RouterLink>
      </nav>

      <div
        v-if="!uiStore.sidebarCollapsed"
        class="sidebar-footer"
      >
        <span>当前阶段</span>
        <strong>A0.5</strong>
      </div>
    </aside>

    <div class="app-content">
      <header class="topbar">
        <div class="topbar-left">
          <button
            class="collapse-button"
            type="button"
            aria-label="收起或展开侧边栏"
            title="收起或展开侧边栏"
            @click="uiStore.toggleSidebar"
          >
            <span aria-hidden="true">☰</span>
          </button>
          <div
            class="breadcrumb"
            aria-label="当前位置"
          >
            <span>KnowledgeScope</span>
            <span
              class="breadcrumb-separator"
              aria-hidden="true"
            >/</span>
            <span>项目概览</span>
          </div>
        </div>
        <div class="topbar-status">
          <span
            class="status-dot"
            aria-hidden="true"
          />
          <span>Phase A0.5 · Web foundation</span>
        </div>
      </header>

      <main class="main-content">
        <slot />
      </main>
    </div>
  </div>
</template>

<style scoped>
.app-shell {
  display: flex;
  min-height: 100vh;
  background: #f4f6f8;
}

.app-sidebar {
  display: flex;
  flex: 0 0 auto;
  flex-direction: column;
  overflow: hidden;
  background: #ffffff;
  border-right: 1px solid #e6e9ee;
}

.brand-block {
  display: flex;
  align-items: center;
  min-height: 84px;
  gap: 12px;
  padding: 0 20px;
  white-space: nowrap;
}

.brand-mark {
  display: grid;
  flex: 0 0 36px;
  width: 36px;
  height: 36px;
  place-items: center;
  color: #ffffff;
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.04em;
  background: #24364b;
  border-radius: 10px;
}

.brand-copy {
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.brand-name {
  color: #1d2a3a;
  font-size: 15px;
  font-weight: 700;
}

.brand-caption {
  color: #8a96a6;
  font-size: 11px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.side-nav {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 18px 12px;
}

.nav-item {
  display: flex;
  align-items: center;
  min-height: 44px;
  gap: 12px;
  padding: 0 12px;
  color: #6f7d8f;
  font-size: 14px;
  font-weight: 600;
  border-radius: 9px;
}

.nav-item:hover {
  color: #24364b;
  background: #f4f7fa;
}

.nav-item.is-active {
  color: #24364b;
  background: #eaf0f6;
}

.nav-icon {
  width: 18px;
  color: #6f8398;
  font-size: 18px;
  line-height: 1;
  text-align: center;
}

.sidebar-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin: auto 16px 20px;
  padding: 13px 14px;
  color: #7c8999;
  font-size: 12px;
  background: #f7f8fa;
  border: 1px solid #edf0f3;
  border-radius: 9px;
}

.sidebar-footer strong {
  color: #24364b;
  font-size: 13px;
}

.app-content {
  display: flex;
  flex: 1;
  min-width: 0;
  flex-direction: column;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  min-height: 64px;
  padding: 0 32px;
  background: #ffffff;
  border-bottom: 1px solid #e6e9ee;
}

.topbar-left,
.topbar-status,
.breadcrumb {
  display: flex;
  align-items: center;
}

.topbar-left {
  gap: 18px;
}

.collapse-button {
  display: grid;
  width: 32px;
  height: 32px;
  padding: 0;
  place-items: center;
  color: #607086;
  font-size: 16px;
  background: transparent;
  border: 1px solid transparent;
  border-radius: 7px;
  cursor: pointer;
}

.collapse-button:hover {
  color: #24364b;
  background: #f2f5f8;
}

.breadcrumb {
  gap: 10px;
  color: #7c8999;
  font-size: 13px;
}

.breadcrumb span:first-child {
  color: #2d3e51;
  font-weight: 650;
}

.breadcrumb-separator {
  color: #c4cbd4;
}

.topbar-status {
  gap: 8px;
  color: #7c8999;
  font-size: 12px;
}

.status-dot {
  width: 7px;
  height: 7px;
  background: #4b9b77;
  border-radius: 50%;
}

.main-content {
  width: min(1180px, 100%);
  margin: 0 auto;
  padding: 38px 42px 56px;
}

@media (max-width: 900px) {
  .topbar {
    padding: 0 20px;
  }

  .main-content {
    padding: 28px 20px 42px;
  }

  .topbar-status {
    display: none;
  }
}
</style>
