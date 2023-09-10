# duplicate the slurm_snowlines_SITE-ID.bash file and save with new site ID

old_id = "SITE-ID"
new_ids = ["01.23635", "01.23649", "01.23664", "01.26738", "01.27103", "02.04403", "02.05157"]

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
