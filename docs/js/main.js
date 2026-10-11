
(function() {
  // ─── Nav scroll state ───
  const nav = document.getElementById('nav');
  let ticking = false;
  function onScroll() {
    if (!ticking) {
      requestAnimationFrame(() => {
        nav.classList.toggle('scrolled', window.scrollY > 8);
        ticking = false;
      });
      ticking = true;
    }
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  // ─── Active section in nav ───
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav-links a[href^="#"]');
  if ('IntersectionObserver' in window) {
    const navObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const id = entry.target.id;
          navLinks.forEach(link => {
            link.classList.toggle('active', link.getAttribute('href') === '#' + id);
          });
        }
      });
    }, { rootMargin: '-40% 0px -55% 0px' });
    sections.forEach(s => navObserver.observe(s));
  }

  // ─── Copy buttons ───
  document.querySelectorAll('.copy-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const text = btn.getAttribute('data-copy');
      const label = btn.querySelector('.copy-label');
      const original = label.textContent;
      const finish = () => {
        label.textContent = 'Copied';
        btn.classList.add('copied');
        setTimeout(() => {
          label.textContent = original;
          btn.classList.remove('copied');
        }, 1800);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(finish).catch(fallback);
      } else {
        fallback();
      }
      function fallback() {
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); } catch(e) {}
        document.body.removeChild(ta);
        finish();
      }
    });
  });

  // ─── Scroll hint gradients for code blocks ───
  function updateScrollHints() {
    document.querySelectorAll('.install-cmd code, .method-cmd code').forEach(el => {
      const hasOverflow = el.scrollWidth > el.clientWidth + 1;
      el.classList.toggle('has-overflow', hasOverflow);

      if (!el._scrollHintHandler) {
        el._scrollHintHandler = () => {
          const atEnd = el.scrollLeft + el.clientWidth >= el.scrollWidth - 1;
          el.classList.toggle('at-end', atEnd);
        };
        el.addEventListener('scroll', el._scrollHintHandler, { passive: true });
      }
      el._scrollHintHandler();
    });
  }

  window.addEventListener('resize', updateScrollHints);
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(updateScrollHints);
  }
  updateScrollHints();

  // ─── Terminal animation ───
  const terminalContent = [
    { text: '─ drea · minimalist agent harness ───────────────────', cls: 'term-banner', delay: 0 },
    { text: '  model   deepseek-ai/DeepSeek-V4-Flash-0731', cls: 'term-meta', delay: 300 },
    { text: '  workdir /home/you/dev/webapp', cls: 'term-meta', delay: 120 },
    { text: '  safety  confirm before commands & writes', cls: 'term-meta', delay: 120 },
    { text: '', cls: 'term-blank', delay: 250 },
    { text: 'Interactive session. Type a task, /help for commands, or Ctrl-D to quit.', cls: 'term-meta', delay: 300 },
    { text: '│─› scaffold a Go CLI that prints the current time', cls: 'term-user', delay: 500 },
    { text: '', cls: 'term-blank', delay: 250 },
    { text: '╭─ assistant', cls: 'term-info', delay: 450 },
    { text: 'I’ll scaffold a small Go CLI and verify it with go test.', cls: 'term-text', delay: 220 },
    { text: '├─→ git_commit "git commit -m \"chore: baseline scaffold\""', cls: 'term-work', delay: 450 },
    { text: '│  committed baseline: add a scratch repo so the work is reversible', cls: 'term-dim', delay: 250 },
    { text: '├─→ read_file main.go', cls: 'term-work', delay: 400 },
    { text: '│  viewing main.go (lines 1–60)', cls: 'term-dim', delay: 250 },
    { text: '├─→ run_command "$ go test ./..."', cls: 'term-work', delay: 400 },
    { text: '│  ok  example.com/clock 0.031s', cls: 'term-dim', delay: 250 },
    { text: '│  [exit code 0]', cls: 'term-dim', delay: 120 },
    { text: '├─→ run_command "$ go build -o clock ./cmd/clock"', cls: 'term-work', delay: 400 },
    { text: '│  [exit code 0]', cls: 'term-dim', delay: 120 },
    { text: '', cls: 'term-blank', delay: 200 },
    { text: '╰─ usage: 1,528 prompt + 312 completion tokens over 3 request(s)', cls: 'term-result', delay: 350 },
    { text: '│─›', cls: 'term-prompt', delay: 250 }
  ];

  const term = document.getElementById('terminal');
  const replayBtn = document.getElementById('term-replay');
  let timeouts = [];
  let played = false;

  function playTerminal() {
    timeouts.forEach(t => clearTimeout(t));
    timeouts = [];
    term.innerHTML = '';
    replayBtn.classList.remove('show');
    let cumulative = 0;
    terminalContent.forEach((line, i) => {
      cumulative += line.delay;
      const t = setTimeout(() => {
        const div = document.createElement('div');
        div.className = 'term-line ' + line.cls;
        div.textContent = line.text === '' ? '\u00A0' : line.text;
        term.appendChild(div);
        term.scrollTop = term.scrollHeight;
        if (i === terminalContent.length - 1) {
          const tc = setTimeout(() => {
            const cursor = document.createElement('span');
            cursor.className = 'term-cursor';
            const lastLine = term.lastElementChild;
            if (lastLine) {
              lastLine.appendChild(cursor);
            } else {
              term.appendChild(cursor);
            }
            term.scrollTop = term.scrollHeight;
            replayBtn.classList.add('show');
          }, 200);
          timeouts.push(tc);
        }
      }, cumulative);
      timeouts.push(t);
    });
  }

  const termWindow = document.getElementById('terminal-window');
  if ('IntersectionObserver' in window) {
    const termObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting && !played) {
          played = true;
          playTerminal();
          termObserver.unobserve(entry.target);
        }
      });
    }, { threshold: 0.25 });
    termObserver.observe(termWindow);
  } else {
    playTerminal();
  }

  replayBtn.addEventListener('click', playTerminal);
})();
