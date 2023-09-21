# duplicate the slurm_snowlines_SITE-ID.bash file and save with new site ID

old_id = "SITE-ID"
new_ids = ["01.16342", "01.17183", "01.17348", "01.17423", "01.15788",
           "01.14883", "01.15769", "01.12425", "01.12635", "01.13696"]

# Load input file
input_fn = "slurm_RGI60-SITE-ID.bash"
with open(input_fn, "r") as input_file:
    content = input_file.read()

# Iterate over new_ids
for new_id in new_ids:

    # Perform the replacement
    new_content = content.replace(old_id, new_id)

    # Define output file name
    output_fn = input_fn.replace(old_id, new_id)

    # Open the input file for writing
    with open(output_fn, "w") as output_file:
        output_file.write(new_content)
    print("Saved new bash file for " + new_id)
