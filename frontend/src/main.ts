import { VueQueryPlugin, QueryClient } from "@tanstack/vue-query";
import { createPinia } from "pinia";
import { createApp } from "vue";
import "element-plus/dist/index.css";

import App from "./App.vue";
import router from "./router";
import "./styles/main.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 30_000,
    },
  },
});

createApp(App)
  .use(router)
  .use(createPinia())
  .use(VueQueryPlugin, { queryClient })
  .mount("#app");
