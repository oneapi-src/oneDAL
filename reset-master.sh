export target_branch="master"
git checkout $target_branch
git remote add intel_daal git@github.com:oneapi-src/oneDAL.git
git remote -v
git fetch --all
git reset --hard intel_daal/master
git push origin $target_branch -f
