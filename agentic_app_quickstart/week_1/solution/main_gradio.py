# Load .env before any other imports
from pathlib import Path
from dotenv import load_dotenv

# Load .env sitting next to this script
load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)

import os

# Load environment variables for telemetry configuration
import asyncio
import pandas as pd
import gradio as gr
from time import time
import matplotlib.pyplot as plt

from agents import Runner, SQLiteSession
from agentic_app_quickstart.week_1.solution.agent import (
    create_analysis_agent,
    create_data_loader_agent,
    create_communication_agent,
)
from agentic_app_quickstart.week_1.solution.hooks import FollowUpHooks
# Phoenix telemetry is handled directly in initialize_system function


# Plotting helper functions
def create_salary_department_chart(df):
    """Create a bar chart of average salary by department"""
    try:
        os.makedirs("plots", exist_ok=True)
        timestamp = int(time())
        filename = f"bar_salary_by_department_{timestamp}.png"
        filepath = os.path.join("plots", filename)
        
        # Group by department and calculate mean salary
        grouped_data = df.groupby("department")["salary"].mean().sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(grouped_data)), grouped_data.values, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title("Average Salary by Department")
        plt.xlabel("Department")
        plt.ylabel("Average Salary ($)")
        plt.xticks(range(len(grouped_data)), grouped_data.index, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return f"\n\n📊 **Bar Chart Generated:** {filename}", filepath
    except Exception as e:
        return f"\n\n⚠️ **Bar chart generation failed:** {str(e)}", None

def create_salary_histogram(df, column_name):
    """Create a histogram of the specified column"""
    try:
        os.makedirs("plots", exist_ok=True)
        timestamp = int(time())
        filename = f"hist_{column_name}_{timestamp}.png"
        filepath = os.path.join("plots", filename)
        
        plt.figure(figsize=(10, 6))
        plt.hist(df[column_name].dropna(), bins=20, edgecolor='black', alpha=0.7)
        plt.title(f"{column_name.title()} Distribution")
        plt.xlabel(column_name.title())
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return f"\n\n📊 **Histogram Generated:** {filename}", filepath
    except Exception as e:
        return f"\n\n⚠️ **Histogram generation failed:** {str(e)}", None

def create_scatter_plot_direct(df, x_column, y_column):
    """Create a scatter plot of x_column vs y_column"""
    try:
        os.makedirs("plots", exist_ok=True)
        timestamp = int(time())
        filename = f"scatter_{x_column}_vs_{y_column}_{timestamp}.png"
        filepath = os.path.join("plots", filename)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(df[x_column], df[y_column], alpha=0.6, edgecolors='black', linewidth=0.5)
        plt.title(f"{x_column.title()} vs {y_column.title()}")
        plt.xlabel(x_column.title())
        plt.ylabel(y_column.title())
        plt.grid(True, alpha=0.3)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return f"\n\n📊 **Scatter Plot Generated:** {filename}", filepath
    except Exception as e:
        return f"\n\n⚠️ **Scatter plot generation failed:** {str(e)}", None


def create_weather_city_temp_chart(df):
    """Create a bar chart of temperature by city"""
    try:
        os.makedirs("plots", exist_ok=True)
        timestamp = int(time())
        filename = f"bar_city_vs_temperature_{timestamp}.png"
        filepath = os.path.join("plots", filename)
        
        # Group by city and calculate mean temperature
        grouped_data = df.groupby("city")["temperature"].mean().sort_values(ascending=False)
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(grouped_data)), grouped_data.values, color='lightcoral', edgecolor='black', alpha=0.7)
        plt.title("Temperature Across Cities")
        plt.xlabel("City")
        plt.ylabel("Temperature (°C)")
        plt.xticks(range(len(grouped_data)), grouped_data.index, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return f"\n\n📊 **Weather Chart Generated:** {filename}", filepath
    except Exception as e:
        return f"\n\n⚠️ **Weather chart generation failed:** {str(e)}", None


def create_weather_histogram(df, column_name):
    """Create a histogram of weather data columns"""
    try:
        os.makedirs("plots", exist_ok=True)
        timestamp = int(time())
        filename = f"hist_{column_name}_{timestamp}.png"
        filepath = os.path.join("plots", filename)
        
        plt.figure(figsize=(10, 6))
        plt.hist(df[column_name].dropna(), bins=20, edgecolor='black', alpha=0.7, color='lightblue')
        plt.title(f"{column_name.title()} Distribution")
        plt.xlabel(column_name.title())
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return f"\n\n📊 **Weather Histogram Generated:** {filename}", filepath
    except Exception as e:
        return f"\n\n⚠️ **Weather histogram generation failed:** {str(e)}", None


# Global session and hooks for persistence across chat
session = None
hooks = None

# Global chat history to maintain conversation
chat_history = []


def initialize_system():
    """Initialize the system once when the app starts"""
    global session, hooks
    
    # .env is already loaded at top; don't call load_dotenv() again here
    
    # Initialize Phoenix telemetry directly here
    try:
        from phoenix.otel import register
        
        phoenix_api_key = os.getenv("PHOENIX_API_KEY")
        if not phoenix_api_key:
            raise ValueError("PHOENIX_API_KEY environment variable is required")
        
        print("🔗 Phoenix endpoint:", os.getenv("PHOENIX_COLLECTOR_ENDPOINT"))
        print("🔑 Phoenix key present:", bool(phoenix_api_key))
        print("🔑 Phoenix key value:", phoenix_api_key[:20] + "..." if phoenix_api_key else "None")
        
        tracer_provider = register(
            project_name=os.getenv("PHOENIX_PROJECT_NAME", "MyProject"),
            endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "https://app.phoenix.arize.com/s/markt/v1/traces"),
            api_key=phoenix_api_key,
            batch=True,
        )
        
        print("✅ Phoenix telemetry initialized")
    except Exception as e:
        print(f"⚠️ Telemetry initialization warning: {e}")
    
    # Simple initialization without complex dependencies
    try:
        session = SQLiteSession(session_id=123)
        hooks = FollowUpHooks()
        print("✅ Session and hooks created successfully")
    except Exception as e:
        print(f"⚠️ Session initialization warning: {e}")
        # Create dummy objects if real ones fail
        session = None
        hooks = None
    
    return "System initialized! You can now ask questions about the data."


async def process_question(question, history):
    """
    Process a question through the three-agent pipeline and return the response
    """
    if not session or not hooks:
        return "Please initialize the system first."
    
    if not question.strip():
        return "Please enter a question."
    
    try:
        # Decide which dataset to load based on the question
        loader_result = await Runner.run(
            starting_agent=create_data_loader_agent(),
            input=question,
            session=session,
        )
        key = str(getattr(loader_result, "final_output", "")).strip().lower()

        path_map = {
            "employees": "data/employee_data.csv",
            "weather": "data/weather_data.csv",
            "sales": "data/sample_sales.csv",
        }

        # Auto-select the best dataset based on AI recommendation
        if key in path_map:
            csv_path = path_map[key]
            dataset_info = f"Data loader selected: {key} -> {csv_path}"
        else:
            # Fallback: try to infer from question content
            question_lower = question.lower()
            if any(word in question_lower for word in ["employee", "salary", "performance", "hr"]):
                key = "employees"
            elif any(word in question_lower for word in ["weather", "temperature", "humidity", "climate"]):
                key = "weather"
            elif any(word in question_lower for word in ["sales", "price", "quantity", "transaction"]):
                key = "sales"
            else:
                key = "employees"  # Default fallback
            csv_path = path_map[key]
            dataset_info = f"Auto-selected dataset: {key} -> {csv_path}"

        # Load CSV into context
        content = pd.read_csv(csv_path)
        
        # Get column information for plotting
        numeric_cols = content.select_dtypes(include="number").columns.tolist()
        categorical_cols = content.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Analyze
        analysis_result = await Runner.run(
            starting_agent=create_analysis_agent(),
            input=question,
            context=content,
            session=session,
        )
        analysis_text = str(getattr(analysis_result, "final_output", "")).strip()

        # Communicate
        final_result = await Runner.run(
            starting_agent=create_communication_agent(),
            input=question,
            context=analysis_text,
            session=session,
        )

        final_response = getattr(final_result, "final_output", "")
        
        # Add dataset info to response
        full_response = f"{dataset_info}\n\n{final_response}"
        
        # Add follow-up suggestions if available
        if hooks.last_suggestions:
            suggestions = "\n\n**Suggested follow-up questions:**\n" + "\n".join([f"• {s}" for s in hooks.last_suggestions[:3]])
            full_response += suggestions
        
        # Check if we should generate plots based on the response
        plot_info = ""
        plot_files = []
        if any(word in final_response.lower() for word in ["plot", "chart", "graph", "visualization", "bar chart", "histogram", "scatter"]):
            try:
                # Generate appropriate plots based on data type and question
                if key == "employees" and "salary" in question.lower() and "department" in question.lower():
                    # Create bar chart for salary by department
                    plot_info, plot_file = create_salary_department_chart(content)
                    if plot_file:
                        plot_files.append(plot_file)
                elif "histogram" in final_response.lower() or "distribution" in final_response.lower():
                    # Create histogram for numeric columns
                    if numeric_cols:
                        plot_info, plot_file = create_salary_histogram(content, numeric_cols[0])
                        if plot_file:
                            plot_files.append(plot_file)
                elif "scatter" in final_response.lower() or "correlation" in final_response.lower():
                    # Create scatter plot for numeric columns
                    if len(numeric_cols) >= 2:
                        plot_info, plot_file = create_scatter_plot_direct(content, numeric_cols[0], numeric_cols[1])
                        if plot_file:
                            plot_files.append(plot_file)
            except Exception as e:
                plot_info = f"\n\n⚠️ **Plot generation failed:** {str(e)}"
        
        # Return response with plot info and plot files
        return full_response + plot_info, plot_files
        
    except Exception as e:
        return f"Error processing question: {str(e)}"


def create_gradio_interface():
    """Create and return the Gradio interface"""
    
    with gr.Blocks(title="Data Analysis Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📊 Data Analysis Agent")
        gr.Markdown("Ask questions about employee, weather, or sales data and get AI-powered analysis.")
        
        # Initialize system automatically in background (non-blocking)
        
        # Chat area using HTML for better image support
        chat_html = gr.HTML(
            value="<div id='chat-container' style='height: 400px; overflow-y: auto; padding: 20px; background: #2d3748; border-radius: 8px;'>"
                  "<div style='text-align: center; color: #a0aec0; margin-top: 150px;'>"
                  "💬 Start a conversation by asking a question about the data</div>"
                  "</div>"
        )
        
        # Input section (sticky bottom)
        with gr.Row():
            question_input = gr.Textbox(
                label="Ask a question about the data",
                placeholder="e.g., What's the average customer salary? (Press Enter to send)",
                lines=2,
                scale=4
            )
            submit_btn = gr.Button("Send", variant="primary", size="sm", scale=1)
        
        # Refresh button below input
        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh Chat", size="sm", variant="secondary")
        

        
        # Simple synchronous initialization
        try:
            initialize_system()
            print("✅ System initialized successfully")
        except Exception as e:
            print(f"⚠️ System initialization warning: {e}")
        
        def build_chat_html():
            """Build chat HTML from current chat history"""
            chat_html_content = "<div id='chat-container' style='height: 400px; overflow-y: auto; padding: 20px; background: #2d3748; border-radius: 8px;'>"
            
            for msg in chat_history:
                # User message
                chat_html_content += f"""
                    <div style='margin-bottom: 20px;'>
                        <div style='background: #4299e1; color: white; padding: 12px; border-radius: 8px; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);'>
                            <strong>👤 You:</strong> {msg['user']}
                        </div>
                """
                
                # AI response or loading indicator
                if msg.get('status') == "loading" or msg['ai'] == "🤖 AI is thinking...":
                    chat_html_content += f"""
                        <div style='background: #1a202c; padding: 20px; border-radius: 8px; margin-bottom: 15px; text-align: center; border: 2px solid #4299e1; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);'>
                            <div style='display: inline-block; width: 40px; height: 40px; border: 4px solid #2d3748; border-top: 4px solid #4299e1; border-radius: 50%; animation: spin 1s linear infinite;'></div>
                            <div style='margin-top: 15px; color: #4299e1; font-size: 18px; font-weight: 600;'>🤖 AI is thinking...</div>
                            <div style='margin-top: 8px; color: #a0aec0; font-size: 14px;'>Processing your question...</div>
                        </div>
                        <style>
                        @keyframes spin {{
                            0% {{ transform: rotate(0deg); }}
                            100% {{ transform: rotate(360deg); }}
                        }}
                        </style>
                    """
                else:
                    chat_html_content += f"""
                        <div style='background: #4a5568; padding: 15px; border-radius: 8px; margin-bottom: 15px; color: white; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);'>
                            <strong style='color: white;'>🤖 AI:</strong> {msg['ai']}
                        </div>
                    """
                
                # Add plots if they exist
                for plot_file in msg['plots']:
                    if plot_file and os.path.exists(plot_file):
                        try:
                            import base64
                            with open(plot_file, "rb") as img_file:
                                img_data = base64.b64encode(img_file.read()).decode()
                            
                            chat_html_content += f"""
                            <div style='background: #4a5568; padding: 15px; border-radius: 8px; margin-bottom: 15px; text-align: center; color: white; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);'>
                                <strong>📊 Generated Plot:</strong><br>
                                <img src="data:image/png;base64,{img_data}" 
                                     alt="Generated Plot" 
                                     style="max-width: 100%; height: auto; border: 2px solid #4299e1; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.3); margin-top: 10px;">
                            </div>
                            """
                        except Exception as e:
                            chat_html_content += f"""
                            <div style='background: #4a5568; padding: 15px; border-radius: 8px; margin-bottom: 15px; color: white; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);'>
                                <strong>📊 Generated Plot:</strong> {os.path.basename(plot_file)} (display failed: {str(e)})
                            </div>
                            """
                
                chat_html_content += "</div>"
            
            chat_html_content += "</div>"
            return chat_html_content

        def handle_submit(question, chat_content):
            if not question.strip():
                return chat_content, ""
            
            # Check if system is ready
            global session, hooks, chat_history
            if session is None or hooks is None:
                return chat_content, ""
            
            # Add user message to chat history immediately
            chat_history.append({
                "user": question,
                "ai": "🤖 AI is thinking...",  # Show loading message immediately
                "plots": [],
                "status": "loading"  # Track processing status
            })
            
            # Build and return the loading state immediately
            loading_html = build_chat_html()
            
            # Process the question directly (async)
            try:
                result = asyncio.run(process_question(question, []))
                
                # Handle response and plots
                if isinstance(result, tuple) and len(result) == 2:
                    response, plot_files = result
                else:
                    response = result
                    plot_files = []
                
                # Auto-generate plots based on the specific question content
                if "plot" in question.lower() and not plot_files:
                    try:
                        # Determine dataset and plot type based on the question
                        question_lower = question.lower()
                        
                        if any(word in question_lower for word in ["temperature", "weather", "city", "temp", "climate"]):
                            # Weather data plots
                            if "city" in question_lower and "temperature" in question_lower:
                                # Load weather data for city vs temperature
                                weather_content = pd.read_csv("data/weather_data.csv")
                                plot_result, plot_file = create_weather_city_temp_chart(weather_content)
                                plot_type = "temperature across cities"
                            else:
                                # Default weather plot
                                weather_content = pd.read_csv("data/weather_data.csv")
                                plot_result, plot_file = create_weather_histogram(weather_content, "temperature")
                                plot_type = "temperature distribution"
                        elif any(word in question_lower for word in ["salary", "employee", "department", "performance"]):
                            # Employee data plots
                            employee_content = pd.read_csv("data/employee_data.csv")
                            if "salary" in question_lower and "department" in question_lower:
                                plot_result, plot_file = create_salary_department_chart(employee_content)
                                plot_type = "average salary by department"
                            elif "salary" in question_lower and "distribution" in question_lower:
                                plot_result, plot_file = create_salary_histogram(employee_content, "salary")
                                plot_type = "salary distribution histogram"
                            elif "performance" in question_lower and "salary" in question_lower:
                                plot_result, plot_file = create_scatter_plot_direct(employee_content, "performance_score", "salary")
                                plot_type = "performance vs salary scatter plot"
                            else:
                                plot_result, plot_file = create_salary_histogram(employee_content, "salary")
                                plot_type = "salary distribution histogram"
                        else:
                            # Default to employee data
                            employee_content = pd.read_csv("data/employee_data.csv")
                            plot_result, plot_file = create_salary_histogram(employee_content, "salary")
                            plot_type = "salary distribution histogram"

                        if plot_file:
                            plot_files.append(plot_file)
                            response += f"\n\n📊 **Generated plot:** {plot_type}"
                        else:
                            response += f"\n\n⚠️ **Plot generation failed:** No plot file was created"
                            
                    except Exception as e:
                        response += f"\n\n⚠️ **Plot generation failed:** {str(e)}"
                        print(f"Plot generation error: {e}")  # Debug logging
                
                # Update the last message in chat history with real response
                if chat_history and chat_history[-1]["user"] == question:
                    chat_history[-1]["ai"] = response
                    chat_history[-1]["plots"] = plot_files
                    chat_history[-1]["status"] = "complete"  # Mark as complete
                
                # Return the final result directly
                final_html = build_chat_html()
                return final_html, ""
                
            except Exception as e:
                print(f"Error processing question: {e}")
                if chat_history and chat_history[-1]["user"] == question:
                    chat_history[-1]["ai"] = f"Error: {str(e)}"
                    chat_history[-1]["plots"] = []
                    chat_history[-1]["status"] = "error"  # Mark as error
                
                # Return error state
                error_html = build_chat_html()
                return error_html, ""
            

        
        # Handle button click with loading indicator
        submit_btn.click(
            fn=handle_submit,
            inputs=[question_input, chat_html],
            outputs=[chat_html, question_input],
            show_progress=False,
            api_name="submit"
        )
        
        # Enable Enter key to submit (no loading on input)
        question_input.submit(
            fn=handle_submit,
            inputs=[question_input, chat_html],
            outputs=[chat_html, question_input],
            show_progress=False,
            api_name="submit_enter"
        )
        
        # Add a refresh button to manually update the chat when needed
        def refresh_chat():
            """Refresh the chat to show any completed responses"""
            return build_chat_html()
        
        # Connect refresh button to function
        refresh_btn.click(
            fn=refresh_chat,
            outputs=chat_html
        )
    
    return demo


if __name__ == "__main__":
    # Create and launch the Gradio interface
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=8003,
        share=False,
        show_error=True
    )
