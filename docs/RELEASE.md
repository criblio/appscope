# Cribl AppScope - Release Process

## Branching & Tagging 

Our process is pretty straight forward. We're focused primarily on developing
the next release and minimizing effort toward maintaining older releases.

* The default branch is `master`.

* Work on defects and feature requests is done on _issue_ branches from the
  default branch named like `bug/1234-name` and `feature/1234-name`,
  respectively. The number prefix is an issue number and the `-name` part
  should align with the issue title. Occasionally, we skip creating an issue
  and omit the number. When these branches are complete, a PR is created, 
  reviewed, and merged back in to the default branch.

* We tag the default branch like `v1.0.0-rc1` for release candidates then 
  `v1.0.0` when it's ready to ship. 

* If a maintenance release (i.e. `v1.0.1`) is needed, we create a `release/1.0`
  branch, apply fixes there as needed, tag that branch `v1.0.1-rc1` then
  `v1.0.1` to sjip it. Later maintenance releases for are built and tagged on
  this branch too. We branch "late" like this since we expect to only do this
  occasionally.

* We have a `staging` branch for use with the website content. The
  documentation team makes changes to the website content on that branch then
  create PRs to merge their changes into the default branch.

## CI/CD

We use GitHub Actions to automate CI/CD tasks when changes are made in the
project's repository. See [`.github/workflows/`](../.github/workflows/) for
details. We build the code and run the unit tests with every push, on every
branch. If the test pass, some additional steps are taken depending on the
branch or tag.

* The `scope` binary and a `scope.tgz` that contains `scope`, `ldscope`,
  `libscope.so` and the default YAML configs are pushed to an AWS-S3 container
  that is exposed publicly at `https://cdn.cribl.io/dl/scope/`. Below that base
  URL we have:

  * `latest` - text file with the latest release number in it; i.e. `0.6.1`
  * `$VERSION/linux/scope`
  * `$VERSION/linux/scope.md5`
  * `$VERSION/linux/scope.tgz`
  * `$VERSION/linux/scope.tgz.md5`

  The `$VERSION` in the examples above is a release tag (without the leading
  `v`, i.e. `1.2.3` or `1.2.3-rc1`), an branch (i.e. `branch/bug/1234-name`) or
  `next` for the default branch.

  The `.md5` files are MD5 checksums of the corresponding files without the
  extension.

* For `v*` tags, we build and push container images to Docker Hub repositories
  at `cribl/scope:version` and `cribl/scope-demo:version`. See
  [`docker/`](../docker/) for details.

* For tags, pushes to the default branch, and pull requests, we run a suite of
  [integration tests](../test/testContainers/)

A separate GitHub workflow handles building and deploying the
[`website/`](../website/) content to be built and deployed to
<https://appscope.dev/> on pushes to the default branch. Pushes to the
`staging` branch do the same but to the staging website at
<https://staging.appscope.dev/>.
