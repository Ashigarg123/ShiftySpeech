import os 
import argparse
import sys

def create_test_file(bona_dir, spoof_dir, save_name):
    with open(save_name, 'w') as out:
        for root, _, files in os.walk(bona_dir):
            for file in files:
                if file.endswith(".wav") or file.endswith(".flac"):
                    utt_pth = os.path.join(root, file)
                    out.write(f"{utt_pth} bonafide\n")
        
        for root, _, files in os.walk(spoof_dir):
            for file in files:
                if file.endswith(".wav") or file.endswith(".flac"):
                    utt_pth = os.path.join(root, file)
                    out.write(f"{utt_pth} spoof\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare test file')
    parser.add_argument('--save_path', type=str, default="test.txt", help="Path to save the test file")
    parser.add_argument('--bona_dir', type=str, help="Path to directory with original/real audio samples")
    parser.add_argument('--spoof_dir', type=str, help="Path to directory with fake/spoof audio samples")

    args = parser.parse_args()
    create_test_file(args.bona_dir, args.spoof_dir, args.save_path)



