import os
import argparse
import subprocess

def process_traj(traj_dir, output_base_dir, start_number, codec, frame_rate, pixel_format):
    # import pdb; pdb.set_trace()
    images_pattern = os.path.join(traj_dir, 'images0', 'im_%d.jpg')
    # Make sure the images folder exists
    if not os.path.exists(os.path.join(traj_dir, 'images0')):
        # print(f"Skipping {traj_dir}: 'images0' folder does not exist.")
        return
    print(f"Processing {traj_dir}...")
    # Extract relevant components to form the output video file name
    relative_path = os.path.relpath(traj_dir, 'jesse_datasets')
    output_file = relative_path.split('datasets/')[1]
    output_file = output_file.split('/')
    output_file = output_file[0] + '/' + output_file[-1] + '.mp4'
    # import pdb; pdb.set_trace()
    # output_file = f"{relative_path.replace('/', '_')}.mp4"
    output_path = os.path.join(output_base_dir, output_file)
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    
    # # Construct the ffmpeg command
    ffmpeg_command = [
        'ffmpeg', '-start_number', str(start_number), '-i', images_pattern,
        '-c:v', codec, '-r', str(frame_rate), '-pix_fmt', pixel_format, output_path
    ]

    print(f"Running ffmpeg command: {' '.join(ffmpeg_command)}")
    # # Run the command
    subprocess.run(ffmpeg_command)

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Process trajectories in jesse_datasets.")
    parser.add_argument('--traj', type=int, help="Specific trajectory number to process.")
    parser.add_argument('--recursive', action='store_true', help="Recursively process all trajectories.")

    args = parser.parse_args()

    # Define the base paths and ffmpeg options
    input_base_dir = '/scr/jzhang96/jesse_datasets'
    output_base_dir = '/scr/jzhang96/jesse_datasets_videos'
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    start_number = 0
    codec = 'libx264'
    frame_rate = 30
    pixel_format = 'yuv420p'

    # Recursively process all traj folders
    if args.recursive:
        for root, dirs, files in os.walk(input_base_dir):
            for dir_name in dirs:
                traj_dir = os.path.join(root, dir_name)
                if 'traj' in dir_name:  # Look for directories containing 'traj'
                    traj_dir = os.path.join(root, dir_name)
                    # for i in range(20):
                    #     video_dir = os.path.join(traj_dir, f'traj{i}')
                    # print(traj_dir)
                        # process_traj(video_dir, output_base_dir, start_number, codec, frame_rate, pixel_format)
                    process_traj(traj_dir, output_base_dir, start_number, codec, frame_rate, pixel_format)

    # Process a specific traj number if provided
    elif args.traj is not None:
        for root, dirs, _ in os.walk(input_base_dir):
            for dir_name in dirs:
                if dir_name == f'traj{args.traj}':
                    traj_dir = os.path.join(root, dir_name)
                    process_traj(traj_dir, output_base_dir, start_number, codec, frame_rate, pixel_format)
                    #return  # Exit after processing the specified trajectory

    else:
        print("Please specify either --traj <number> or --recursive.")

if __name__ == "__main__":
    main()
