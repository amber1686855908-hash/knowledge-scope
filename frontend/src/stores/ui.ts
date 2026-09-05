import { defineStore } from "pinia";

export const useUiStore = defineStore("ui", {
  state: () => ({
    sidebarCollapsed: false,
  }),
  actions: {
    toggleSidebar(): void {
      this.sidebarCollapsed = !this.sidebarCollapsed;
    },
  },
});
