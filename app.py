import base64
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import gradio as gr
from firecrawl import FirecrawlApp, ScrapeOptions
from PIL import Image
from xai_sdk import Client
from xai_sdk.chat import image, system, tool, tool_result, user

# Initialize clients with connection pooling
client = Client(
    api_key=os.getenv("XAI_API_KEY"),
    timeout=3600,
)
fc = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))


def get_medicine_info_fast(name: str) -> Dict:
    """Super fast medicine info fetcher with aggressive optimization"""
    try:
        # Ultra-fast search with minimal timeout and no markdown scraping
        results = fc.search(
            query=f"{name} medicine",  # Simplified query
            limit=1,
            scrape_options=ScrapeOptions(formats=["markdown"], timeout=30000),
            tbs="qdr:w",
        )
        snippet = results.data[0] if results.data else {}
        return {
            "name": name,
            "info_markdown": snippet.get("markdown", snippet.get("description", "Basic medicine information available")),
            "url": snippet.get("url", "N/A"),
            "description": snippet.get("description", f"{name} - Medicine information from search results"),
            "status": "success",
        }
    except Exception as e:
        # Return quick fallback data instead of error
        return {
            "name": name,
            "info_markdown": f"## {name}\n\nCommon medicine. Please consult your pharmacist for detailed information.",
            "url": "N/A",
            "description": f"{name} - Please consult healthcare provider for usage and dosage information",
            "status": "fallback",
        }

def get_multiple_medicines_concurrent(
    medicine_names: List[str], max_workers: int = 5
) -> List[Dict]:
    """Fetch multiple medicine info concurrently"""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_medicine = {
            executor.submit(get_medicine_info_fast, name): name
            for name in medicine_names
        }
        for future in as_completed(future_to_medicine):
            try:
                result = future.result(timeout=30)
                results.append(result)
            except Exception as e:
                medicine_name = future_to_medicine[future]
                results.append(
                    {
                        "name": medicine_name,
                        "info_markdown": "Timeout or error",
                        "url": "N/A",
                        "description": f"Error: {str(e)}",
                        "status": "error",
                    }
                )
    return results


# Tool definitions
tool_definitions = [
    tool(
        name="get_medicine_info_fast",
        description="Fetch markdown info, URL, and description for a medicine via Firecrawl (optimized)",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the medicine"},
            },
            "required": ["name"],
        },
    ),
    tool(
        name="get_multiple_medicines_concurrent",
        description="Fetch info for multiple medicines concurrently",
        parameters={
            "type": "object",
            "properties": {
                "medicine_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of medicine names",
                },
                "max_workers": {
                    "type": "integer",
                    "description": "Maximum concurrent workers (default: 5)",
                    "default": 5,
                },
            },
            "required": ["medicine_names"],
        },
    ),
]

tools_map = {
    "get_medicine_info_fast": get_medicine_info_fast,
    "get_multiple_medicines_concurrent": get_multiple_medicines_concurrent,
}


def encode_image_from_bytes(image_bytes) -> str:
    """Encode image bytes to base64 string"""
    return base64.b64encode(image_bytes).decode("utf-8")


def get_image_mime_type(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.format == "JPEG":
            return "image/jpeg"
        elif img.format == "PNG":
            return "image/png"
        else:
            return None
    except Exception:
        return None


def analyze_prescription_streaming(file_bytes):
    """Analyze prescription with streaming progress updates using generator"""
    try:
        yield "📈 **Starting Analysis...**\n\nValidating uploaded image..."

        image_bytes = file_bytes
        mime_type = get_image_mime_type(image_bytes)
        if mime_type is None:
            yield "Error: Uploaded file is not a valid JPG or PNG image."
            return

        yield "✅ **Image Validated**\n\nPreparing image for AI analysis..."

        encoded_img = encode_image_from_bytes(image_bytes)
        image_data_url = f"data:{mime_type};base64,{encoded_img}"

        yield "🤖 **Connecting to Grok-4 AI**\n\nInitializing chat session..."

        # Create chat session
        chat = client.chat.create(
            model="grok-4",
            tools=tool_definitions,
            tool_choice="auto",
        )

        # Enhanced system prompt for better extraction
        chat.append(
            system(
                "You are MedGuide AI. Extract ALL medicine names from the prescription image. "
                "If you find multiple medicines, use get_multiple_medicines_concurrent to fetch "
                "all information at once for faster processing. For single medicine, use get_medicine_info_fast. "
                "Create a comprehensive markdown report."
            )
        )

        # User provides the prescription image
        chat.append(
            user(
                "Extract all medicine names from this prescription and get their details efficiently.",
                image(image_url=image_data_url),
            )
        )

        yield "🧠 **Image understanding**\n\nExtracting medicine names from prescription..."

        # Initial model call
        response = chat.sample()
        chat.append(response)

        # Execute tool calls if any
        if response.tool_calls:
            tool_results = []
            for i, tc in enumerate(response.tool_calls, 1):
                func_name = tc.function.name
                func_args = json.loads(tc.function.arguments)

                args_display = json.dumps(func_args, indent=2)
                current_progress = f"🛠️ **Tool Call {i}/{len(response.tool_calls)}:**\n\n**Function:** `{func_name}`\n\n**Arguments:**\n```json\n{args_display}\n```\n\n⏳ Fetching data from Firecrawl API..."
                yield current_progress

                result = tools_map[func_name](**func_args)
                tool_results.append((func_name, func_args, result))
                chat.append(tool_result(json.dumps(result)))

                # Show API response in JSON format with limited content
                if isinstance(result, list):
                    # Multiple medicines response - create summary for each
                    summary_result = []
                    for item in result:
                        summary_item = {
                            "name": item.get('name', 'Unknown'),
                            "status": item.get('status', 'unknown'),
                            "url": item.get('url', 'N/A')[:80] + "..." if len(item.get('url', '')) > 80 else item.get('url', 'N/A'),
                            "info_markdown": item.get('info_markdown', '')[:100] + "..." if len(item.get('info_markdown', '')) > 100 else item.get('info_markdown', 'N/A'),
                            "description": item.get('description', '')[:100] + "..." if len(item.get('description', '')) > 100 else item.get('description', 'N/A')
                        }
                        summary_result.append(summary_item)
                    
                    json_display = json.dumps(summary_result, indent=2)
                    summary = f"✅ **Firecrawl API Response ({len(result)} medicines):**\n```json\n{json_display}\n```"
                else:
                    # Single medicine response
                    summary_item = {
                        "name": result.get('name', 'Unknown'),
                        "status": result.get('status', 'unknown'),
                        "url": result.get('url', 'N/A')[:80] + "..." if len(result.get('url', '')) > 80 else result.get('url', 'N/A'),
                        "info_markdown": result.get('info_markdown', '')[:100] + "..." if len(result.get('info_markdown', '')) > 100 else result.get('info_markdown', 'N/A'),
                        "description": result.get('description', '')[:100] + "..." if len(result.get('description', '')) > 100 else result.get('description', 'N/A')
                    }
                    
                    json_display = json.dumps(summary_item, indent=2)
                    summary = f"✅ **Firecrawl API Response:**\n```json\n{json_display}\n```"
                
                yield summary + "\n\n"

            yield "🤔 **Grok 4 is Thinking**\n\nAI is thinking and will soon start generating the report..."

            # Request final formatted report
            chat.append(
                user(
                    "Create a comprehensive markdown report with H2 heading for each medicine that contains: Description, "
                    "Typical Duration, Price Information, and Purchase Link"
                )
            )

            # Generate final report with streaming
            accumulated_content = ""
            try:
                # Use streaming to get the final report
                stream = chat.stream()
                for stream_chunk in stream:
                    # XAI SDK returns tuples with (Response, Chunk)
                    for item in stream_chunk:
                        if hasattr(item, 'choices') and item.choices:
                            choice = item.choices[0]
                            if hasattr(choice, 'content') and choice.content:
                                text_content = choice.content
                                accumulated_content += text_content
                                # Yield the accumulated content so far for real-time streaming
                                yield f"📝 **Live Report Generation:**\n\n{accumulated_content}"
                
                # Final yield with complete content
                if accumulated_content:
                    yield accumulated_content
                else:
                    # Fallback if streaming fails
                    final = chat.sample()
                    yield final.content
                    
            except Exception as stream_error:
                # If streaming fails, fallback to regular sample
                try:
                    final = chat.sample()
                    yield final.content
                except Exception as e:
                    yield f"Error generating final report: {str(e)}"
        else:
            yield response.content

    except Exception as e:
        yield f"Error analyzing prescription: {str(e)}"


# Gradio interface (Blocks version)
def main():
    # Processing state to prevent multiple requests
    processing_state = {"is_processing": False}

    with gr.Blocks(theme=gr.themes.Base()) as demo:
        gr.Markdown(
            """
            # 🏥MedGuide AI: Prescription Analyzer
            Upload a photo of your medical prescription. The AI will extract all medicine names and provide a detailed markdown report including description, duration, price, and purchase links.
            
            > Powered by 💬Grok-4 AI and 🔥Firecrawl API.
            """
        )
        with gr.Row():
            with gr.Column():
                file_input = gr.Image(
                    label="Upload Prescription Image",
                    type="pil",
                    height=400,  # Set image component height
                    width=400,  # Set image component width
                    format="png",
                )
                analyze_btn = gr.Button("Analyze Prescription", variant="primary")
            with gr.Column():
                report_output = gr.Markdown(
                    label="📋 Final Report",
                )
                time_output = gr.Markdown(label="⏱️ Processing Time")

                # Collapsible logs section
                with gr.Accordion(
                    "🔍 Processing Logs (Click to view details)", open=False
                ):
                    logs_output = gr.Markdown(
                        label="Processing Details",
                        value="Logs will appear here during processing...",
                    )

        def analyze_with_streaming_progress(image):
            """Analyze prescription with streaming progress updates"""
            # Prevent multiple concurrent requests
            if processing_state["is_processing"]:
                yield (
                    "⚠️ **Already Processing:** Please wait for the current analysis to complete.",
                    "Another request is already being processed.",
                    "Please wait for the current analysis to finish before starting a new one.",
                    gr.update(
                        interactive=True, value="Analyze Prescription"
                    ),  # Keep button enabled for this message
                )
                return

            processing_state["is_processing"] = True
            start_time = time.time()

            try:
                if image is None:
                    yield (
                        "❌ **Error:** Please upload an image first.",
                        "No image provided.",
                        "Logs will appear here during processing...",
                        gr.update(
                            interactive=True, value="Analyze Prescription"
                        ),  # Re-enable button on error
                    )
                    return

                # Convert PIL image to bytes
                buf = io.BytesIO()
                image.save(
                    buf,
                    format=image.format
                    if hasattr(image, "format") and image.format
                    else "PNG",
                )
                image_bytes = buf.getvalue()

                # Accumulate all logs to show complete process
                all_logs = []
                final_markdown_report = ""

                # Use the streaming generator
                for progress_update in analyze_prescription_streaming(image_bytes):
                    elapsed = time.time() - start_time

                    # Check if this looks like the final markdown report
                    # (contains medicine headings, descriptions, or is longer content)
                    # Also detect streaming report format
                    is_final_report = (
                        progress_update.startswith("# ")
                        or progress_update.startswith("## ")
                        or (
                            "Medicine" in progress_update and len(progress_update) > 100
                        )
                        or "Description" in progress_update
                        or "Price" in progress_update
                        or "Duration" in progress_update
                        or ("📝 **Live Report Generation:**" in progress_update and len(progress_update) > 200)
                    )
                    
                    # If it's a streaming report, extract the actual content
                    if "📝 **Live Report Generation:**" in progress_update:
                        # Extract content after the streaming header
                        content_start = progress_update.find("📝 **Live Report Generation:**\n\n") + len("📝 **Live Report Generation:**\n\n")
                        if content_start < len(progress_update):
                            actual_content = progress_update[content_start:]
                            # Check if the actual content looks like a report
                            if (
                                actual_content.startswith("# ")
                                or actual_content.startswith("## ")
                                or "Medicine" in actual_content
                                or "Description" in actual_content
                                or len(actual_content) > 50
                            ):
                                # Show the streaming content in the main report area
                                yield (
                                    actual_content,
                                    f"Streaming... {elapsed:.1f}s elapsed",
                                    "\n\n".join(all_logs),
                                    gr.update(
                                        interactive=False, value="⏳ Processing..."
                                    ),
                                )
                                continue
                    
                    if is_final_report:
                        # This is the final report - show in main report, keep logs in accordion
                        final_markdown_report = progress_update
                        final_logs = (
                            "\n\n".join(all_logs)
                            if all_logs
                            else "Processing completed successfully!"
                        )
                        yield (
                            final_markdown_report,
                            f"✅ Completed in {elapsed:.2f} seconds",
                            final_logs,
                            gr.update(
                                interactive=True, value="Analyze Prescription"
                            ),  # Re-enable button
                        )
                        return
                    else:
                        # This is a process log - accumulate and show in logs section
                        all_logs.append(progress_update)
                        # Show processing message in main area, detailed logs in accordion
                        processing_message = "👨‍⚕️ **Processing in progress...**\n\nAnalyzing prescription image and fetching medicine information.\n\n*Check the Processing Logs section below for detailed step-by-step progress.*"
                        yield (
                            processing_message,
                            f"Processing... {elapsed:.1f}s elapsed",
                            "\n\n".join(all_logs),
                            gr.update(
                                interactive=False, value="⏳ Processing..."
                            ),  # Keep button disabled during processing
                        )

                # If we somehow don't detect the final report, show the last update
                if not final_markdown_report:
                    elapsed = time.time() - start_time
                    final_logs = (
                        "\n\n".join(all_logs) if all_logs else "Processing completed."
                    )
                    # Check if the last update could be the report
                    if all_logs and len(all_logs[-1]) > 50:
                        yield (
                            all_logs[-1],
                            f"✅ Completed in {elapsed:.2f} seconds",
                            final_logs,
                            gr.update(
                                interactive=True, value="Analyze Prescription"
                            ),  # Re-enable button
                        )
                    else:
                        yield (
                            "Analysis completed. Please check the processing logs for details.",
                            f"✅ Completed in {elapsed:.2f} seconds",
                            final_logs,
                            gr.update(
                                interactive=True, value="Analyze Prescription"
                            ),  # Re-enable button
                        )
            except Exception as e:
                # Handle any unexpected errors
                elapsed = time.time() - start_time
                yield (
                    f"❌ **Error during processing:** {str(e)}",
                    f"Error after {elapsed:.2f} seconds",
                    "An unexpected error occurred during processing.",
                    gr.update(
                        interactive=True, value="Analyze Prescription"
                    ),  # Re-enable button on error
                )
            finally:
                # Always reset processing state
                processing_state["is_processing"] = False

        def show_initial_processing(image):
            """Show initial processing message and disable button"""
            if image is None:
                return (
                    "❌ **Error:** Please upload an image first.",
                    "No image provided.",
                    "Logs will appear here during processing...",
                    gr.update(
                        interactive=True, value="Analyze Prescription"
                    ),  # Keep enabled if no image
                )
            return (
                "🚀 **Processing Started!**\n\nInitializing analysis...",
                "Processing...",
                "Initializing...",
                gr.update(
                    interactive=False, value="⏳ Processing..."
                ),  # Disable button during processing
            )

        # Event handlers
        analyze_btn.click(
            show_initial_processing,
            inputs=[file_input],
            outputs=[report_output, time_output, logs_output, analyze_btn],
            queue=False,
        )

        analyze_btn.click(
            analyze_with_streaming_progress,
            inputs=[file_input],
            outputs=[report_output, time_output, logs_output, analyze_btn],
            queue=True,
        )

        # Medical Disclaimer Section (Collapsible)
        with gr.Accordion("⚠️ Medical Disclaimer - Please Read", open=False):
            gr.Markdown(
                """
                **IMPORTANT NOTICE:** This application is for **informational purposes only** and should **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment.
                
                ### 🚨 Key Points:
                - **Always consult your doctor** or qualified healthcare provider with any questions about medications
                - **Verify all information** with your pharmacist or healthcare provider before taking any medication
                - **Do not rely solely** on this AI analysis for medical decisions
                - **AI may make errors** - always double-check medicine names, dosages, and instructions
                - **Emergency situations** require immediate medical attention, not AI analysis
                
                ### 📋 This Tool:
                - Provides general information about medicines found in prescription images
                - Uses AI to extract text and search for publicly available information
                - May not be 100% accurate in reading prescriptions or providing information
                - Should be used as a **supplementary tool only**
                
                ### 🩺 Always Remember:
                **Your healthcare provider knows your medical history and current condition best. When in doubt, always consult with them directly.**
                
                ---
                
                *By using this application, you acknowledge that you understand these limitations and agree to use the information responsibly.*
                """
            )

    demo.launch()


if __name__ == "__main__":
    main()
