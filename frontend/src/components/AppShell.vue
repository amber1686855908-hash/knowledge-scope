<script setup lang="ts">
import { RouterLink } from "vue-router";

import { useUiStore } from "../stores/ui";

const uiStore = useUiStore();
</script>

<template>
  <div
    class="app-shell"
    :class="{ 'is-sidebar-collapsed': uiStore.sidebarCollapsed }"
  >
    <aside
      class="app-sidebar"
    >
      <div class="brand-block">
        <div class="brand-header">
          <RouterLink
            to="/knowledge-bases"
            class="brand-link"
            aria-label="KnowledgeScope 知识库"
          >
            <span
              class="brand-mark"
              aria-hidden="true"
            >K</span>
            <div
              v-if="!uiStore.sidebarCollapsed"
              class="brand-copy"
            >
              <span class="brand-name">KnowledgeScope</span>
              <span class="brand-caption">行业文档</span>
            </div>
          </RouterLink>
          <button
            class="collapse-button"
            type="button"
            aria-label="收起或展开侧边栏"
            title="收起或展开侧边栏"
            @click="uiStore.toggleSidebar"
          >
            <span aria-hidden="true">☰</span>
          </button>
        </div>
      </div>

      <nav
        class="side-nav"
        aria-label="主导航"
      >
        <RouterLink
          to="/knowledge-bases"
          class="nav-item"
          active-class="is-active"
          title="知识库"
        >
          <span
            class="nav-icon"
            aria-hidden="true"
          >▦</span>
          <span
            v-if="!uiStore.sidebarCollapsed"
            class="nav-label"
          >知识库</span>
        </RouterLink>
      </nav>
    </aside>

    <div class="app-content">
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
  background: var(--ks-bg);
}

.app-sidebar {
  display: flex;
  width: 232px;
  flex: 0 0 232px;
  flex-direction: column;
  overflow: hidden;
  background: var(--ks-surface);
  border-right: 1px solid var(--ks-border);
  transition: width var(--ks-duration-normal) var(--ks-ease-out),
    flex-basis var(--ks-duration-normal) var(--ks-ease-out);
}

.is-sidebar-collapsed .app-sidebar {
  width: 72px;
  flex-basis: 72px;
}

.brand-block {
  min-height: 80px;
  padding: 0 16px;
  white-space: nowrap;
}

.brand-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  min-height: 80px;
  gap: 8px;
}

.brand-link {
  display: flex;
  min-width: 0;
  align-items: center;
  min-height: 48px;
  gap: 12px;
  padding: 0 4px;
  border-radius: var(--ks-radius-sm);
}

.brand-link:hover .brand-name {
  color: var(--ks-accent-strong);
}

.brand-mark {
  display: grid;
  flex: 0 0 34px;
  width: 34px;
  height: 34px;
  place-items: center;
  color: var(--ks-surface);
  font-size: 15px;
  font-weight: 750;
  letter-spacing: -0.04em;
  background: var(--ks-ink);
  border-radius: 9px;
}

.brand-copy {
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.brand-name {
  color: var(--ks-ink);
  font-size: 15px;
  font-weight: 720;
  letter-spacing: -0.02em;
}

.brand-caption {
  color: var(--ks-muted);
  font-size: 11px;
}

.side-nav {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 16px 12px;
}

.nav-item {
  display: flex;
  align-items: center;
  min-height: 42px;
  gap: 12px;
  padding: 0 12px;
  color: var(--ks-muted);
  font-size: 14px;
  font-weight: 620;
  border-radius: var(--ks-radius-sm);
  transition: color var(--ks-duration-fast) var(--ks-ease-out),
    background-color var(--ks-duration-fast) var(--ks-ease-out);
}

.nav-item:hover {
  color: var(--ks-ink);
  background: var(--ks-surface-subtle);
}

.nav-item.is-active {
  color: var(--ks-accent-strong);
  background: var(--ks-accent-soft);
}

.nav-icon {
  width: 18px;
  color: currentColor;
  font-size: 17px;
  line-height: 1;
  text-align: center;
}

.app-content {
  display: flex;
  flex: 1;
  min-width: 0;
  flex-direction: column;
}

.is-sidebar-collapsed .brand-header {
  justify-content: center;
  flex-direction: column;
  gap: 4px;
  padding: 8px 0;
}

.is-sidebar-collapsed .brand-link {
  flex: 0 0 auto;
}

.collapse-button {
  display: grid;
  width: 32px;
  height: 32px;
  padding: 0;
  place-items: center;
  color: var(--ks-muted);
  font-size: 16px;
  background: transparent;
  border: 1px solid transparent;
  border-radius: var(--ks-radius-sm);
  cursor: pointer;
  transition: color var(--ks-duration-fast) var(--ks-ease-out),
    background-color var(--ks-duration-fast) var(--ks-ease-out),
    transform var(--ks-duration-fast) var(--ks-ease-out);
}

.collapse-button:hover {
  color: var(--ks-ink);
  background: var(--ks-surface);
}

.main-content {
  width: min(1240px, 100%);
  margin: 0 auto;
  padding: 40px 40px 64px;
}

@media (max-width: 900px) {
  .main-content {
    padding: 32px 24px 48px;
  }
}

@media (max-width: 720px) {
  .app-sidebar,
  .is-sidebar-collapsed .app-sidebar {
    width: 68px;
    flex-basis: 68px;
  }

  .brand-block {
    padding: 0 12px;
  }

  .brand-header {
    justify-content: center;
    flex-direction: column;
    gap: 4px;
    padding: 8px 0;
  }

  .brand-link {
    justify-content: center;
    padding: 0;
  }

  .brand-copy,
  .nav-label {
    display: none;
  }

  .nav-item {
    justify-content: center;
    padding: 0;
  }

  .main-content {
    padding: 28px 18px 42px;
  }
}
</style>
