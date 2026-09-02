import { defineConfig } from 'vitepress'
import { readdirSync, existsSync } from 'node:fs'
import { join } from 'node:path'
import type { DefaultTheme } from 'vitepress'

const REPO = 'https://github.com/SolerHo/GPU-Mode-Notes'
const ROOT = process.cwd()

/** 从文件夹名中提取序号，用于排序（Lecture001 → 1） */
function lectureNo(name: string): number {
  const m = name.match(/\d+/)
  return m ? Number.parseInt(m[0], 10) : 0
}

/**
 * 扫描根目录下的课程文件夹（如 Lecture001 / Bonus_Lecture01），
 * 要求文件夹内有 README.md（即已写笔记）。新增课程文件夹后侧边栏自动出现。
 */
function findLectures(): { dir: string; hasEn: boolean }[] {
  return readdirSync(ROOT, { withFileTypes: true })
    .filter(
      (d) =>
        d.isDirectory() &&
        /^(Lecture|Bonus_Lecture)\d+/.test(d.name) &&
        existsSync(join(ROOT, d.name, 'README.md')),
    )
    .map((d) => ({ dir: d.name, hasEn: existsSync(join(ROOT, d.name, 'README_EN.md')) }))
    .sort((a, b) => lectureNo(a.dir) - lectureNo(b.dir))
}

/** 生成侧边栏：课程总览 + 各讲笔记 */
function buildSidebar(): DefaultTheme.SidebarItem[] {
  const items: DefaultTheme.SidebarItem[] = [{ text: '🏠 课程总览', link: '/' }]
  for (const { dir, hasEn } of findLectures()) {
    const item: DefaultTheme.SidebarItem = {
      text: dir.replace('_Lecture', ' ').replace('Lecture', 'Lecture '),
      link: `/${dir}/`,
    }
    if (hasEn) {
      item.children = [{ text: 'English Notes', link: `/${dir}/README_EN` }]
    }
    items.push(item)
  }
  return items
}

export default defineConfig({
  lang: 'zh-CN',
  title: 'GPU-Mode Lecture 学习笔记',
  description: 'GPU MODE 系列课程学习笔记（CUDA / PyTorch / Triton / CUTLASS）',
  base: '/GPU-Mode-Notes/', // GitHub Pages 项目站点部署在子路径下
  cleanUrls: true,
  lastUpdated: true,

  markdown: {
    image: { lazyLoading: true },
  },

  /**
   * 让 GitHub 与 VitePress 站点共用同一份 Markdown，无需维护两份：
   *   README.md             → 站点首页 /          （内容即仓库 README）
   *   Lecture001/README.md  → /Lecture001/         （目录首页，URL 干净）
   * 其余 .md（如 README_EN.md）保持原样路由。
   */
  rewrites: {
    'README.md': 'index.md',
    ':lecture/README.md': ':lecture/index.md',
  },

  themeConfig: {
    nav: [
      { text: '📋 课程总览', link: '/' },
      { text: 'GitHub', link: REPO },
    ],
    sidebar: buildSidebar(),
    socialLinks: [{ icon: 'github', link: REPO }],
    outline: { label: '本页目录', level: [2, 3] },
    docFooter: { prev: '上一篇', next: '下一篇' },
    lastUpdated: { text: '最后更新于', formatOptions: { dateStyle: 'short', timeStyle: 'short' } },
    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '菜单',
    darkModeSwitchLabel: '主题',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式',
    search: { provider: 'local' },
  },
})
