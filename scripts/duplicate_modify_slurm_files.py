# duplicate the slurm_snowlines_SITE-ID.bash file and save with new site ID

old_id = "SITE-ID"
new_ids = ["01.00032", "01.00033", "01.00037", "01.00038", "01.00046", "01.00312", "01.00566", "01.00570", "01.00576",
           "01.00675", "01.01104", "01.01151", "01.01390", "01.01524", "01.01733", "01.03594", "01.03622", "01.03861",
           "01.04375", "01.04624", "01.06268", "01.06279", "01.06722", "01.08155", "01.08174", "01.08246", "01.08248",
           "01.08262", "01.08288", "01.08296", "01.08302", "01.08336", "01.08353", "01.08389", "01.08395", "01.08403",
           "01.08412", "01.08427", "01.09148", "01.09162", "01.09216", "01.09411", "01.09639"]

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
