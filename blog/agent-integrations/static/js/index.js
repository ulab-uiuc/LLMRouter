
const zh = document.documentElement.lang.startsWith('zh');
// Separate HTML pages keep both languages readable without JavaScript.
document.querySelectorAll('.masthead nav a').forEach(link => {
  link.addEventListener('click', () => { link.hash = location.hash; });
});
document.querySelectorAll('.code').forEach(block => {
  const button = document.createElement('button');
  button.type = 'button'; button.className = 'copy';
  button.textContent = zh ? '复制' : 'Copy';
  button.setAttribute('aria-label', zh ? '复制代码' : 'Copy code');
  button.setAttribute('aria-live','polite');
  button.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(block.querySelector('code').textContent);
      button.textContent = zh ? '已复制' : 'Copied';
    } catch {
      const range = document.createRange(); range.selectNodeContents(block.querySelector('code'));
      const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
      button.textContent = zh ? '已选中，请手动复制' : 'Selected; copy manually';
    }
    setTimeout(() => { button.textContent = zh ? '复制' : 'Copy'; }, 2500);
  });
  block.append(button);
});
