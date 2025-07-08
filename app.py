import gradio as gr
import json
import os
import uuid
import threading
from test_travelplanner_interactive import collect_inital_codes, pipeline, GPT_response

def query_to_json(query):
    with open('prompts/query_to_json.txt', 'r') as file:
        query_to_json_prompt = file.read()
    query_to_json_prompt += query
    response = GPT_response(query_to_json_prompt, 'gpt-4')
    return json.loads(response)

def travel_planner(query_text):
    # 1. Convert query to JSON
    yield "Converting your query to a structured format...", "", "", ""
    try:
        query_json = query_to_json(query_text)
        yield f"""Query converted to JSON: 
{json.dumps(query_json, indent=2)}""", "", "", ""
    except Exception as e:
        yield f"Error converting query to JSON: {e}", "", "", ""
        return

    # 2. Setup for planning
    mode = "gradio"
    user_mode = "all_yes" 
    index = str(uuid.uuid4())
    
    output_path = f'output/{mode}/{user_mode}/{index}/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 3. Run planning process in a separate thread
    yield "Generating travel plan...", "", "", ""
    
    try:
        # We need to run the planning in a thread to be able to yield updates
        # to the Gradio interface. However, the pipeline function is complex and
        # not originally designed for this. For this example, we will call it
        # directly and the UI will be blocked until it finishes.
        # A more advanced implementation would require refactoring the pipeline.

        collect_inital_codes(query_json, mode, index)
        
        # The pipeline function writes files but doesn't return the plan directly.
        # We will try to read the output from the files it creates.
        # This is a workaround based on the current structure of the script.
        
        # Create a thread to run the pipeline
        pipeline_thread = threading.Thread(target=pipeline, args=(query_json, mode, user_mode, index))
        pipeline_thread.start()

        # Wait for the pipeline to finish, and show progress
        suggestion_file_path = os.path.join(output_path, 'plans', 'suggestions.txt')
        plan_file_path = os.path.join(output_path, 'plans', 'plan.txt')

        while pipeline_thread.is_alive():
            if os.path.exists(suggestion_file_path):
                with open(suggestion_file_path, 'r') as f:
                    suggestions = f.read()
                yield "The planner is working...", "", suggestions, ""
            gr.time.sleep(5) # Use gr.time.sleep in gradio loops

        pipeline_thread.join()


        if os.path.exists(plan_file_path):
            with open(plan_file_path, 'r') as f:
                plan = f.read()
            yield "Plan generated successfully!", plan, "", ""
        else:
            # If no plan, read the suggestions/logs
            suggestions = ""
            if os.path.exists(suggestion_file_path):
                with open(suggestion_file_path, 'r') as f:
                    suggestions = f.read()
            
            final_info_prompt = ""
            # find the latest info_suggest_prompt
            info_prompt_files = sorted([f for f in os.listdir(os.path.join(output_path, 'plans')) if f.startswith('info_suggest_prompt')])
            if info_prompt_files:
                with open(os.path.join(output_path, 'plans', info_prompt_files[-1]), 'r') as f:
                    final_info_prompt = f.read()

            yield "Could not generate a plan. See logs for details.", "", suggestions, final_info_prompt

    except Exception as e:
        yield f"An error occurred: {e}", "", "", ""


iface = gr.Interface(
    fn=travel_planner,
    inputs=gr.Textbox(lines=5, label="Your Travel Query"),
    outputs=[
        gr.Textbox(label="Status"),
        gr.Textbox(label="Final Plan"),
        gr.Textbox(label="Suggestions Log"),
        gr.Textbox(label="Planner Interaction Log")
    ],
    title="Interactive Travel Planner",
    description="Enter your travel requirements and get a detailed plan. If your query is too constrained, the planner will interact to find a feasible solution."
)

if __name__ == "__main__":
    iface.launch()
