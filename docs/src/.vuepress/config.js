const { description, name } = require('../../package')



module.exports = {
  /**
   * Ref：https://v1.vuepress.vuejs.org/config/#title
   */
  title: 'PrivateLoRA',
  /**
   * Ref：https://v1.vuepress.vuejs.org/config/#description
   */
  description: description,


  base: `/`,
  extendMarkdown(md) {
    md.set({ html: true })
    md.use(require("markdown-it-katex"))
  },
  /**
   * Extra tags to be injected to the page HTML `<head>`
   *
   * ref：https://v1.vuepress.vuejs.org/config/#head
   */
  head: [
    ['meta', { name: 'theme-color', content: '#3eaf7c' }],
    ['meta', { name: 'apple-mobile-web-app-capable', content: 'yes' }],
    ['meta', { name: 'apple-mobile-web-app-status-bar-style', content: 'black' }],
    ['link', { rel: 'stylesheet', href: 'https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.5.1/katex.min.css' }],
    ['link', { rel: 'stylesheet', href: 'https://cdn.jsdelivr.net/github-markdown-css/2.2.1/github-markdown.css' }]

  ],

  /**
   * Theme configuration, here is the default theme configuration for VuePress.
   *
   * ref：https://v1.vuepress.vuejs.org/theme/default-theme-config.html
   */
  themeConfig: {
    repo: '',
    editLinks: false,
    docsDir: '',
    editLinkText: '',
    lastUpdated: true,
    navbar: false,
    nav: [
      {
        text: 'Paper',
        link: 'https://arxiv.org/abs/2311.14030',
      },
      {
        text: 'Code',
        link: 'https://github.com/alipay/private_llm'
      },
      {
        text: 'Demo',
        link: 'https://github.com/alipay/private_llm'
      }
    ],
    // displayAllHeaders: true,
    // sidebar: [
    //   "/"
    // ]
    // sidebar: {
    //   // '/guide/': [
    //   //   {
    //   //     title: 'Guide',
    //   //     collapsable: false,
    //   //     children: [
    //   //       '',
    //   //       'using-vue',
    //   //     ]
    //   //   }
    //   // ],
    // },

  },

  /**
   * Apply plugins，ref：https://v1.vuepress.vuejs.org/zh/plugin/
   */
  plugins: [
    '@vuepress/plugin-back-to-top',
    '@vuepress/plugin-medium-zoom',
  ]
}
