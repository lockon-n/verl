# step 1, git clone,  enter, and go to correct branch
git clone https://$GITHUB_TOKEN@github.com/lockon-n/verl.git
cd verl
git checkout main

# step 2, run something
python playground/test.py

bash playground/test.sh