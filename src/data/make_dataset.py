df = pd.read_csv(file_path) # read in csv file as a DataFrame
df.to_csv(file_path, index=False) # save a DataFrame as csv file

# read all csv under a folder and concatenate them into a big dataframe
path = r'path'

# flat
all_files = glob.glob(os.path.join(path, "*.csv"))

# or recursively
all_files = [os.path.join(root, filename)
             for root, dirnames, filenames in os.walk(path)
             for filename in fnmatch.filter(filenames, '*.csv')]
             
df = pd.concat((pd.read_csv(f) for f in all_files))