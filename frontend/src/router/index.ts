import { createRouter, createWebHistory } from "vue-router";

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: "/",
      redirect: { name: "knowledge-bases" },
    },
    {
      path: "/knowledge-bases",
      name: "knowledge-bases",
      component: () => import("../views/KnowledgeBaseListView.vue"),
      meta: {
        title: "知识库",
      },
    },
    {
      path: "/knowledge-bases/:id",
      name: "knowledge-base-detail",
      component: () => import("../views/KnowledgeBaseDetailView.vue"),
      meta: {
        title: "知识库",
      },
    },
    {
      path: "/:pathMatch(.*)*",
      name: "not-found",
      component: () => import("../views/NotFoundView.vue"),
      meta: {
        title: "页面不存在",
      },
    },
  ],
});

export default router;
