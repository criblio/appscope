# Cribl AppScope - Docs Website

The content in the `website/` folder is used to generate the public-facing website for AppScope end-users at <https://appscope.dev/> and the staging version at <https://staging.appscope.dev/>. Our automated build systems deploy to these sites for commits to the `master` and `staging` branches, respectively. We use [Gatsby](https://www.gatsbyjs.com/) to compile to content here into static content that we then deploy to an AWS S3 bucket that is served up via the URLs mentioned earlier.

The [`deploy.sh`](./deploy.sh) script is run by the [`website` workflow](../.gitlab/workflows/../../.github/workflows/website.yml) to handle the deployment on commits.

See the Gatsby docs for complete details but to spin up a local environment for content editors, it should be a matter of:
1. Install `node` with `brew install node` on MacOS or `apt install nodejs` on Ubuntu. Keep it updated running `brew upgrade node` or `apt update`/`apt upgrade` occasionally.
2. Install local dependencies in the project with `npm install`.
3. Run the local development server with `npm run develop`
4. Access the local development version of the website at [`http://127.0.0.1:8000/`](https://127.0.0.1:8000/).

> Using Node 16 or higher breaks some if this. Works with 14.
