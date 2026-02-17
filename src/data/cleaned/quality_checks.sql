SELECT COUNT(*) as total_rows 
FROM `financial-distress-ew.foresight_cleaned.final_training_set`;

-- 2. Check for Nulls in Critical Columns (Target: 0)
SELECT COUNT(*) as missing_assets
FROM `financial-distress-ew.foresight_cleaned.final_training_set`
WHERE Assets IS NULL;

-- 3. Check Years (Target: Should see 2010-2024)
SELECT fiscal_year, COUNT(*) as count
FROM `financial-distress-ew.foresight_cleaned.final_training_set`
GROUP BY fiscal_year
ORDER BY fiscal_year DESC
LIMIT 5;