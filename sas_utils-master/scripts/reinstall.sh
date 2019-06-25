DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"

cd $DIR

pip3.6 uninstall sas_utils -y
pip3.6 install . --user
