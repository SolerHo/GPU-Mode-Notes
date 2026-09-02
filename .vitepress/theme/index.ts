import DefaultTheme from 'vitepress/theme'
import Viewer from 'viewerjs'
import 'viewerjs/dist/viewer.css'
import './style.css'
import { nextTick, onMounted, watch } from 'vue'
import { useRoute } from 'vitepress'
import type { Theme } from 'vitepress'

// 笔记图片点击 → 居中弹窗（lightbox）查看
let viewer: Viewer | null = null

function initViewer() {
  viewer?.destroy()
  const container = document.querySelector('.vp-doc')
  if (!container) return
  viewer = new Viewer(container as HTMLElement, {
    selector: 'img',
    // 不依赖 transitionend 事件（某些环境下 CSS 过渡时长被覆盖为 0，
    // 导致 shown() 永远不执行、弹窗打开后无图片渲染）
    transition: false,
    navbar: false, // 不显示底部缩略图条
    title: false, // 不显示文件名标题
    toolbar: {
      zoomIn: 1,
      zoomOut: 1,
      oneToOne: 1,
      reset: 1,
      prev: 1,
      next: 1,
      rotateLeft: 1,
      rotateRight: 1,
    },
  })
}

export default {
  extends: DefaultTheme,
  setup() {
    const route = useRoute()
    onMounted(() => initViewer())

    // SPA 切换页面后，内容区 DOM 被重建，需要重新绑定
    watch(
      () => route.path,
      () => nextTick(() => initViewer()),
    )
  },
} satisfies Theme
