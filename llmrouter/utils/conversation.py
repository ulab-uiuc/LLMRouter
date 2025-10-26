"""
Conversation processing utilities for LLMRouter scripts
"""

import json

def extract_user_prompt(conversation, turn=1):
    """Extract user prompt from conversation based on turn number"""
    try:
        if isinstance(conversation, str):
            conversation = json.loads(conversation)
        elif isinstance(conversation, list):
            pass  # Already parsed
        else:
            return ""
        
        # Find the user message for the specified turn
        user_messages = [msg for msg in conversation if msg.get('role') == 'user']
        if len(user_messages) >= turn:
            return user_messages[turn - 1].get('content', '')
        return ""
    except Exception as e:
        print(f"Error extracting user prompt for turn {turn}: {e}")
        return ""

def extract_model_response(conversation, turn=1):
    """Extract model response from conversation based on turn number"""
    try:
        if isinstance(conversation, str):
            conversation = json.loads(conversation)
        elif isinstance(conversation, list):
            pass  # Already parsed
        else:
            return ""
        
        # Find the assistant message for the specified turn
        assistant_messages = [msg for msg in conversation if msg.get('role') == 'assistant']
        if len(assistant_messages) >= turn:
            return assistant_messages[turn - 1].get('content', '')
        return ""
    except Exception as e:
        print(f"Error extracting model response for turn {turn}: {e}")
        return ""

def aggregate_preferences_by_query(data, turn_filter=None):
    """
    Aggregate all pairwise preferences for each query and create overall rankings
    
    For each query:
    - Collect all model pairs and their preferences
    - Filter by turn if specified
    - Create a preference matrix
    - Calculate win rates for each model
    - Models with any wins get score 1.0, others get 0.0
    """
    print("Aggregating preferences by query...")
    
    # Group by query
    query_groups = {}
    for idx, sample in enumerate(data):
        try:
            # Check turn filter
            if turn_filter is not None and sample.get('turn') != turn_filter:
                continue
                
            # Extract user prompt based on turn
            turn = sample.get('turn', 1)
            user_prompt_a = extract_user_prompt(sample['conversation_a'], turn)
            user_prompt_b = extract_user_prompt(sample['conversation_b'], turn)
            
            # Use the first non-empty prompt as the query
            user_prompt = user_prompt_a if user_prompt_a else user_prompt_b
            
            if not user_prompt:
                continue
                
            # Extract model responses based on turn
            model_a_response = extract_model_response(sample['conversation_a'], turn)
            model_b_response = extract_model_response(sample['conversation_b'], turn)
            
            if not model_a_response or not model_b_response:
                continue
            
            # Use query as key for grouping
            if user_prompt not in query_groups:
                query_groups[user_prompt] = {
                    'query': user_prompt,
                    'models': {},
                    'preferences': [],
                    'turn': turn
                }
            
            # Store model responses
            model_a = sample['model_a']
            model_b = sample['model_b']
            query_groups[user_prompt]['models'][model_a] = model_a_response
            query_groups[user_prompt]['models'][model_b] = model_b_response
            
            # Store preference - convert winner to model names
            winner = sample.get('winner', None)
            if winner is not None:
                # winner can be 'model_a', 'model_b', or 'tie'
                if winner == 'model_a':
                    winner_model = model_a
                elif winner == 'model_b':
                    winner_model = model_b
                else:  # tie
                    winner_model = None
                
                query_groups[user_prompt]['preferences'].append({
                    'model_a': model_a,
                    'model_b': model_b,
                    'response_a': model_a_response,
                    'response_b': model_b_response,
                    'winner': winner_model,
                    'judge': sample.get('judge', 'unknown'),
                    'question_id': sample.get('question_id', f'mtbench_{idx}')
                })
                
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    print(f"Found {len(query_groups)} unique queries")
    if turn_filter is not None:
        print(f"Filtered to turn {turn_filter} only")
    return query_groups

def calculate_model_scores(query_groups):
    """
    Calculate evaluation scores for each model in each query group
    
    For each query:
    - Count wins for each model
    - Models with any wins get score 1.0
    - Models with no wins get score 0.0
    """
    print("Calculating model scores...")
    
    converted_data = []
    
    for query, group_data in query_groups.items():
        models = group_data['models']
        preferences = group_data['preferences']
        turn = group_data.get('turn', 1)
        
        # Count wins for each model
        model_wins = {model: 0 for model in models.keys()}
        total_comparisons = {model: 0 for model in models.keys()}
        
        for pref in preferences:
            if pref['winner'] is not None:
                model_wins[pref['winner']] += 1
            # Count total comparisons for each model
            total_comparisons[pref['model_a']] += 1
            total_comparisons[pref['model_b']] += 1
        
        # Calculate scores: 1.0 if any wins, 0.0 if no wins
        model_scores = {}
        for model in models.keys():
            if model_wins[model] > 0:
                model_scores[model] = 1.0
            else:
                model_scores[model] = 0.0
        
        # Create samples for each model
        for model, response in models.items():
            # Get all other responses for choices
            other_responses = [resp for m, resp in models.items() if m != model]
            
            sample = {
                'task_name': 'mt_bench',
                'query': query,
                'gt': response,
                'metric': 'preference_score',
                'choices': [response] + other_responses,
                'question_id': f"mtbench_{hash(query) % 100000}_{model}_t{turn}",
                'model_name': model,
                'preference_score': model_scores[model],
                'wins': model_wins[model],
                'total_comparisons': total_comparisons[model],
                'turn': turn
            }
            converted_data.append(sample)
    
    print(f"Created {len(converted_data)} samples from {len(query_groups)} queries")
    return converted_data
