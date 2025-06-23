from PIL import Image
import os

def process_gif(input_path, output_path):
    # Read the input GIF
    img = Image.open(input_path)
    
    # Get dimensions
    width, height = img.size
    mid_height = height // 2
    
    # Lists to store frames
    combined_frames = []
    
    # Process each frame
    try:
        while True:
            # Copy the current frame
            current = img.copy()
            
            # Split the frame into top and bottom halves
            top_half = current.crop((0, 0, width, mid_height))
            bottom_half = current.crop((0, mid_height, width, height))
            
            # Create a new wider image with the halves side by side
            new_frame = Image.new('RGB', (width * 2, mid_height))
            new_frame.paste(top_half, (0, 0))
            new_frame.paste(bottom_half, (width, 0))
            
            # Store the new frame
            combined_frames.append(new_frame)
            
            # Move to next frame
            img.seek(img.tell() + 1)
            
    except EOFError:
        pass  # End of frames
    
    # Save the new GIF
    if combined_frames:
        combined_frames[0].save(
            output_path,
            save_all=True,
            append_images=combined_frames[1:],
            duration=img.info.get('duration', 100),
            loop=0
        )

if __name__ == "__main__":
    input_gif = "assets/rsfm.gif"
    output_gif = "assets/rsfm_recombined.gif"
    
    if not os.path.exists(input_gif):
        print(f"Error: Input file {input_gif} not found")
    else:
        process_gif(input_gif, output_gif)
        print(f"Successfully created {output_gif}")
