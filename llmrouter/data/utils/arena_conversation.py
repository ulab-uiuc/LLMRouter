"""
Arena conversation processing utilities for LLMRouter scripts
"""

import json

def extract_user_prompt(conversation_json):
    """Extract user prompt from OpenAI conversation format"""
    try:
        if isinstance(conversation_json, str):
            conversation = json.loads(conversation_json)
        else:
            conversation = conversation_json
        
        # Find the first user message
        for message in conversation:
            if message.get('role') == 'user':
                return message.get('content', '')
        return ""
    except Exception as e:
        print(f"Error extracting user prompt: {e}")
        return ""

def extract_model_response(conversation_json):
    """Extract model response from OpenAI conversation format"""
    try:
        if isinstance(conversation_json, str):
            conversation = json.loads(conversation_json)
        else:
            conversation = conversation_json
        
        # Find the first assistant message
        for message in conversation:
            if message.get('role') == 'assistant':
                return message.get('content', '')
        return ""
    except Exception as e:
        print(f"Error extracting model response: {e}")
        return ""

def aggregate_preferences_by_query(data):
    """
    Aggregate all pairwise preferences for each query and create overall rankings
    
    For each query:
    - Collect all model pairs and their preferences
    - Create a preference matrix
    - Calculate win rates for each model
    - Models with any wins get score 1.0, others get 0.0
    """
    print("Aggregating preferences by query...")
    
    # First, let's check the dataset structure
    if len(data) > 0:
        print(f"Sample keys: {list(data[0].keys())}")
        print(f"First sample: {data[0]}")
    
    # Group by query
    query_groups = {}
    processed_count = 0
    error_count = 0
    
    for idx, sample in enumerate(data):
        try:
            # Check if the required keys exist
            if 'conversation_a' not in sample or 'conversation_b' not in sample:
                print(f"Sample {idx} missing conversation keys: {list(sample.keys())}")
                error_count += 1
                continue
                
            # Extract user prompt
            user_prompt_a = extract_user_prompt(sample['conversation_a'])
            user_prompt_b = extract_user_prompt(sample['conversation_b'])
            user_prompt = user_prompt_a if user_prompt_a else user_prompt_b
            
            if not user_prompt:
                error_count += 1
                continue
                
            # Extract model responses
            model_a_response = extract_model_response(sample['conversation_a'])
            model_b_response = extract_model_response(sample['conversation_b'])
            
            if not model_a_response or not model_b_response:
                error_count += 1
                continue
            
            # Use query as key for grouping
            if user_prompt not in query_groups:
                query_groups[user_prompt] = {
                    'query': user_prompt,
                    'models': {},
                    'preferences': []
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
                    'model_1': model_a,
                    'model_2': model_b,
                    'response_1': model_a_response,
                    'response_2': model_b_response,
                    'winner': winner_model
                })
            
            processed_count += 1
                
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            error_count += 1
            continue
    
    print(f"Processed {processed_count} samples successfully, {error_count} errors")
    print(f"Found {len(query_groups)} unique queries")
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
        
        # Count wins for each model
        model_wins = {model: 0 for model in models.keys()}
        total_comparisons = {model: 0 for model in models.keys()}
        
        for pref in preferences:
            if pref['winner'] is not None:
                model_wins[pref['winner']] += 1
            # Count total comparisons for each model
            total_comparisons[pref['model_1']] += 1
            total_comparisons[pref['model_2']] += 1
        
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
                'task_name': 'chatbot_arena',
                'query': query,
                'gt': response,
                'metric': 'preference_score',
                'choices': [response] + other_responses,
                'question_id': f"arena_{hash(query) % 100000}_{model}",
                'model_name': model,
                'preference_score': model_scores[model],
                'wins': model_wins[model],
                'total_comparisons': total_comparisons[model]
            }
            converted_data.append(sample)
    
    print(f"Created {len(converted_data)} samples from {len(query_groups)} queries")
    return converted_data
