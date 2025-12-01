/**
 * LLM-powered search integration for MkDocs Material
 */

(function() {
    'use strict';

    const API_ENDPOINT= 'https://llm-router-website.vercel.app/';

    async function searchDocs(query) {
    const response = await fetch(`${API_URL}/search`, {
        method: 'POST',
        headers: {
        'Content-Type': 'application/json',
        },
        body: JSON.stringify({
        query: query,
        n_results: 5
        })
    });
    
    const data = await response.json();
    return data;
    }
    
    // Wait for DOM to be ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    function init() {
        console.log('AI Search: Initializing...');
        addAISearchButton();
        createAISearchModal();
    }

    function addAISearchButton() {
        // Wait for search form to be available
        const checkSearchForm = setInterval(() => {
            const searchForm = document.querySelector('.md-search__form');
            if (searchForm) {
                clearInterval(checkSearchForm);
                console.log('AI Search: Found search form');
                
                // Create AI search button
                const aiButton = document.createElement('button');
                aiButton.type = 'button';
                aiButton.className = 'md-search__icon md-icon ai-search-btn';
                aiButton.innerHTML = `
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
                        <path fill="currentColor" d="M21,11C21,16.55 17.16,21.74 12,23C6.84,21.74 3,16.55 3,11V5L12,1L21,5V11M12,21C15.75,20 19,15.54 19,11.22V6.3L12,3.18L5,6.3V11.22C5,15.54 8.25,20 12,21M10,17V15H8V13H10V11H8V9H10V7H14V9H12V11H14V13H12V15H14V17H10Z" />
                    </svg>
                `;
                aiButton.title = 'Ask AI Assistant (Ctrl+K)';
                aiButton.setAttribute('aria-label', 'Open AI search');
                aiButton.style.cssText = 'cursor: pointer; padding: 0.4rem; border: none; background: transparent;';

                // Add click handler
                aiButton.addEventListener('click', function(e) {
                    e.preventDefault();
                    console.log('AI Search: Button clicked');
                    const searchInput = document.querySelector('[data-md-component="search-query"]');
                    const query = searchInput ? searchInput.value : '';
                    openAISearchModal(query);
                });

                searchForm.appendChild(aiButton);
                console.log('AI Search: Button added to search form');
            }
        }, 100);

        // Stop checking after 5 seconds
        setTimeout(() => clearInterval(checkSearchForm), 5000);

        // Add keyboard shortcut (Ctrl+K or Cmd+K)
        document.addEventListener('keydown', function(e) {
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                openAISearchModal();
            }
        });
    }

    function createAISearchModal() {
        const modal = document.createElement('div');
        modal.id = 'ai-search-modal';
        modal.className = 'ai-search-modal';
        modal.innerHTML = `
            <div class="ai-search-modal-content">
                <div class="ai-search-header">
                    <h2>
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
                            <path fill="currentColor" d="M21,11C21,16.55 17.16,21.74 12,23C6.84,21.74 3,16.55 3,11V5L12,1L21,5V11M12,21C15.75,20 19,15.54 19,11.22V6.3L12,3.18L5,6.3V11.22C5,15.54 8.25,20 12,21M10,17V15H8V13H10V11H8V9H10V7H14V9H12V11H14V13H12V15H14V17H10Z" />
                        </svg>
                        Claude Documentation Assistant
                    </h2>
                    <button class="ai-search-close" aria-label="Close">×</button>
                </div>
                <div class="ai-search-input-container">
                    <input 
                        type="text" 
                        id="ai-search-input" 
                        placeholder="Ask Claude anything about the documentation..."
                        autocomplete="off"
                    />
                    <button id="ai-search-submit" class="ai-search-submit-btn">
                        <span class="search-icon">Ask Claude</span>
                        <span class="loading-icon" style="display: none;">
                            <svg class="spinner" viewBox="0 0 50 50" width="20" height="20">
                                <circle cx="25" cy="25" r="20" fill="none" stroke="white" stroke-width="5"></circle>
                            </svg>
                        </span>
                    </button>
                </div>
                <div id="ai-search-results" class="ai-search-results"></div>
            </div>
        `;

        document.body.appendChild(modal);

        // Add event listeners
        const closeBtn = modal.querySelector('.ai-search-close');
        closeBtn.addEventListener('click', closeAISearchModal);

        modal.addEventListener('click', function(e) {
            if (e.target === modal) {
                closeAISearchModal();
            }
        });

        const input = modal.querySelector('#ai-search-input');
        const submitBtn = modal.querySelector('#ai-search-submit');

        input.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                performAISearch();
            }
        });

        submitBtn.addEventListener('click', performAISearch);

        // ESC key to close
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape' && modal.classList.contains('active')) {
                closeAISearchModal();
            }
        });
    }

    function openAISearchModal(initialQuery = '') {
        const modal = document.getElementById('ai-search-modal');
        const input = document.getElementById('ai-search-input');
        
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
        
        if (initialQuery) {
            input.value = initialQuery;
        }
        
        setTimeout(() => input.focus(), 100);
    }

    function closeAISearchModal() {
        const modal = document.getElementById('ai-search-modal');
        modal.classList.remove('active');
        document.body.style.overflow = '';
    }

    async function performAISearch() {
        const input = document.getElementById('ai-search-input');
        const resultsDiv = document.getElementById('ai-search-results');
        const submitBtn = document.getElementById('ai-search-submit');
        const query = input.value.trim();

        if (!query) {
            return;
        }

        submitBtn.querySelector('.search-icon').style.display = 'none';
        submitBtn.querySelector('.loading-icon').style.display = 'inline-block';
        submitBtn.disabled = true;

        resultsDiv.innerHTML = '<div class="ai-search-loading">Claude is thinking...</div>';

        try {
            const response = await fetch(`${API_ENDPOINT}/search`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    n_results: 5,
                    model: 'meta/llama-3.1-405b-instruct',
                    max_tokens: 2000
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail?.error || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            displayResults(data);

        } catch (error) {
            console.error('Search error:', error);
            resultsDiv.innerHTML = `
                <div class="ai-search-error">
                    <p>❌ Sorry, something went wrong with Claude.</p>
                    <p class="error-detail">${error.message}</p>
                    <p>Please try again or use the regular search.</p>
                </div>
            `;
        } finally {
            submitBtn.querySelector('.search-icon').style.display = 'inline-block';
            submitBtn.querySelector('.loading-icon').style.display = 'none';
            submitBtn.disabled = false;
        }
    }

    function displayResults(data) {
        const resultsDiv = document.getElementById('ai-search-results');
        
        const answerHtml = markdownToHtml(data.answer);
        
        let sourcesHtml = '';
        if (data.sources && data.sources.length > 0) {
            sourcesHtml = `
                <div class="ai-search-sources">
                    <h3>📚 Sources</h3>
                    <ul>
                        ${data.sources.map(source => `
                            <li>
                                <a href="${source.url}" onclick="closeAISearchModal()">
                                    ${source.title}
                                </a>
                            </li>
                        `).join('')}
                    </ul>
                </div>
            `;
        }

        let tokenInfo = '';
        if (data.tokens_used) {
            tokenInfo = `
                <div class="token-info">
                    <span title="Input tokens">${data.tokens_used.input_tokens} in</span>
                    <span title="Output tokens">${data.tokens_used.output_tokens} out</span>
                    <span title="Total tokens">${data.tokens_used.total_tokens} total</span>
                </div>
            `;
        }

        resultsDiv.innerHTML = `
            <div class="ai-search-answer">
                <div class="answer-header">
                    <span class="answer-badge">✨ Claude Answer</span>
                    <span class="model-badge">${data.model_used}</span>
                    ${tokenInfo}
                </div>
                <div class="answer-content">${answerHtml}</div>
            </div>
            ${sourcesHtml}
        `;
    }

    function markdownToHtml(markdown) {
        return markdown
            .replace(/#{3}\s(.+)/g, '<h3>$1</h3>')
            .replace(/#{2}\s(.+)/g, '<h2>$1</h2>')
            .replace(/#{1}\s(.+)/g, '<h1>$1</h1>')
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.+?)\*/g, '<em>$1</em>')
            .replace(/`(.+?)`/g, '<code>$1</code>')
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            .replace(/^(.+)/, '<p>$1')
            .replace(/(.+)$/, '$1</p>');
    }

    window.closeAISearchModal = closeAISearchModal;
})();
