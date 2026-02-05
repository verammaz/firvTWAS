#!/bin/bash
# Check which annotation jobs completed successfully
# Success is determined by having the expected last two lines in .out files

LOG_DIR="logs/annotate"
OUT_PATTERN="annotate_T*.out"

echo "Checking annotation job completion status..."
echo "=========================================="
echo ""

# Find all .out files
OUT_FILES=$(ls -1 ${LOG_DIR}/${OUT_PATTERN} 2>/dev/null | sort)
TOTAL_JOBS=$(echo "${OUT_FILES}" | wc -l)

if [ ${TOTAL_JOBS} -eq 0 ]; then
    echo "No annotation .out files found matching pattern: ${LOG_DIR}/${OUT_PATTERN}"
    exit 1
fi

echo "Total annotation jobs found: ${TOTAL_JOBS}"
echo ""

# Track successful and failed jobs
SUCCESSFUL=()
FAILED=()
MISSING_OUT=()

# Expected last two lines
EXPECTED_LINE1="Total genes:"
EXPECTED_LINE2="Genes with mapped variants:"

for out_file in ${OUT_FILES}; do
    # Extract job identifier (e.g., annotate_Train.12835478_3.out -> Train.12835478_3)
    job_id=$(basename "${out_file}" .out | sed 's/annotate_//')
    
    if [ ! -f "${out_file}" ]; then
        MISSING_OUT+=("${job_id}")
        continue
    fi
    
    # Check last two lines
    if [ -s "${out_file}" ]; then
        # Get last two lines
        last_line=$(tail -n 1 "${out_file}")
        second_last_line=$(tail -n 2 "${out_file}" | head -n 1)
        
        # Check if they match expected pattern
        if [[ "${second_last_line}" == ${EXPECTED_LINE1}* ]] && \
           [[ "${last_line}" == ${EXPECTED_LINE2}* ]]; then
            SUCCESSFUL+=("${job_id}")
        else
            FAILED+=("${job_id}")
            echo "FAILED: ${job_id}"
            echo "  Last two lines:"
            echo "    $(echo "${second_last_line}" | head -c 100)"
            echo "    $(echo "${last_line}" | head -c 100)"
        fi
    else
        FAILED+=("${job_id} (empty .out file)")
    fi
done

echo ""
echo "=========================================="
echo "Summary:"
echo "=========================================="
echo "Successful: ${#SUCCESSFUL[@]} / ${TOTAL_JOBS}"
echo "Failed:     ${#FAILED[@]} / ${TOTAL_JOBS}"

if [ ${#MISSING_OUT[@]} -gt 0 ]; then
    echo "Missing .out: ${#MISSING_OUT[@]}"
fi

echo ""

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed jobs:"
    printf '%s\n' "${FAILED[@]}" | sed 's/^/  - /'
    echo ""
fi

if [ ${#SUCCESSFUL[@]} -eq ${TOTAL_JOBS} ] && [ ${#MISSING_OUT[@]} -eq 0 ]; then
    echo "✓ All jobs completed successfully!"
    exit 0
else
    echo "✗ Some jobs failed or are incomplete."
    exit 1
fi
