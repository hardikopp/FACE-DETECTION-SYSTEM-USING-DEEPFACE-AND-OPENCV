(function () {
    const body = document.body;
    const btn = document.createElement('button');
    btn.style.position = 'fixed';
    btn.style.bottom = '18px';
    btn.style.right = '18px';
    btn.style.padding = '8px 12px';
    btn.style.borderRadius = '18px';
    btn.style.border = 'none';
    btn.style.cursor = 'pointer';
    btn.style.zIndex = 1000;

    let theme = localStorage.getItem('theme') || 'dark';
    apply(theme);

    btn.onclick = function () {
        theme = theme === 'dark' ? 'light' : 'dark';
        apply(theme);
        localStorage.setItem('theme', theme);
    };
    document.body.appendChild(btn);

    function apply(t) {
        if (t === 'dark') {
            document.body.classList.add('dark');
            btn.innerText = '☀️ Light';
            btn.style.background = 'rgba(255,255,255,0.12)';
            btn.style.color = '#fff';
        } else {
            document.body.classList.remove('dark');
            document.body.style.background = '#f6f7fb';
            btn.innerText = '🌙 Dark';
            btn.style.background = 'rgba(0,0,0,0.06)';
            btn.style.color = '#000';
        }
    }
})();
