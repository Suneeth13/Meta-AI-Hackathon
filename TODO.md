# Fix HF Validation: inference.py at Root

## Steps (from approved plan)

### [ ] Step 1: Copy key files from Submission/ to root
copy Submission\inference.py .
copy Submission\baseline.py .
copy Submission\models.py .
copy Submission\client.py .
copy Submission\requirements.txt .
xcopy Submission\server server /E /I
copy Submission\Dockerfile .

### [ ] Step 2: Update root inference.py imports/paths for root server/
(Adjust if needed)

### [ ] Step 3: Test locally
.venv\Scripts\activate & python inference.py
(Expect scores print)

### [ ] Step 4: Git commit on new branch
git checkout -b blackboxai/fix-inference-root
git add .
git commit -m "Fix: Add inference.py to repo root for HF validation"
git push -u origin blackboxai/fix-inference-root
gh pr create --title "Fix HF validation" --body "Adds required files to root"

### [ ] Step 5: Verify HF Spaces validation passes
(Rebuild/deploy)

### [ ] Step 6: attempt_completion

