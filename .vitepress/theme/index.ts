import DefaultTheme from 'vitepress/theme'
import mediumZoom from 'medium-zoom'
import { nextTick, onMounted, watch } from 'vue'
import { useRoute } from 'vitepress'
import type { Theme } from 'vitepress'

// 图片点击放大（medium-zoom）
let zoom: ReturnType<typeof mediumZoom> | null = null

function initZoom() {
  zoom?.destroy()
  zoom = mediumZoom('.vp-doc img', {
    background: 'var(--vp-c-bg)',
  })
}

export default {
  extends: DefaultTheme,
  setup() {
    const route = useRoute()
    onMounted(() => initZoom())

    // SPA 切换页面后重新绑定图片
    watch(
      () => route.path,
      () => nextTick(() => initZoom()),
    )
  },
} satisfies Theme
