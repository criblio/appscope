# Cribl AppScope - Release Process

## Branching & Tagging 

Our process is pretty straight forward.

* The default branch is `master`.

* Work on defects and feature requests is done on _issue_ branches from the
  default branch named like `bug/1234-name` and `feature/1234-name`,
  respectively. The number prefix is an issue number and the `-name` part
  should align with the issue title. Occasionally, we skip creating an issue
  and omit the number though this is frowned upon now.

* We use [Projects](https://github.com/criblio/appscope/projects/) to track
  issues assigned to releases. When assigned issues are done and merged, we
  create a release branch named like `release/1.0` from the default branch.
  Note only the major and minor numbers for the release are in the branch name.

* We tag release branches like `v1.0.0-rc1` for release candidates. Any fixes
  needed get committed to the release branch and additional `-rc` tags are
  created until it's ready to go. We then tag it with `v1.0.0` and ship it.

* If maintenance releases (i.e. `v1.0.1`, `v1.0.2`, etc.) are needed, fixes are
  committed to the release branch and the tagging process above is repeated.

* Optional `hotfix/...` branches can be created from a release branch for more
  complicated maintenance releases or when work needs to be done in parallel.

It would be nice if we could put some controls in place at GitHub to enforce
the branch and tag naming scheme. In the meantime, please follow along.

> TODO: We have a `staging` branch so the [`website/`](../website/) content can be
  pushed to separate production and staging instances. Details of how that's to
  be used have not been worked into this write-up yet.

> TODO: Cherry-picking master commits out to release branches?

## CI/CD

We use GitHub Actions to automate CI/CD tasks when changes are made in the
project's repository. See [`.github/workflows/`](../.github/workflows/) for
details. The gist is below.

* We build the code and run the unit tests with every push, on every branch.

* If the unit tests pass, the built executables and configs are pushed to an
  AWS-S3 container and exposed at `https://cdn.cribl.io/dl/scope/...` 

  * `.../latest` the latest release number; i.e. `0.6.1`
  * `.../$VERSION/linux/scope`
  * `.../$VERSION/linux/scope.md5`
  * `.../$VERSION/linux/scope.tgz`
  * `.../$VERSION/linux/scope.tgz.md5`
  * `.../$VERSION/linux/scope.tgz.md5`

  The `$VERSION` in the examples above is a release tag (without the leading
  `v`, i.e. `1.2.3` or `1.2.3-rc1`), an branch (i.e. `branch/1234-name`) or
  `next` for the default branch.

  The `.tgz` files contain `scope`, `ldscope`, `libscope.so` and the default
  configs.  The `.md5` files are MD5 checksums of the corresponding files
  without the extension.

* Also if the unit tests pass, but only for `v*` tags, we build and push
  container images to Docker Hub repositories at `cribl/scope:version` and
  `cribl/scope-demo:version`. See [`docker/`](../docker/) for details.

* We run a suite of [integration tests](../test/testContainers/) on the default
  branch nightly and can manually trigger this when needed. Pushes to a release
  branch should trigger these tests too but that's not in place yet.

* Pushes to the default and `staging` branches trigger the
  [`website/`](../website/) content to be built and deployed to
  <https://appscope.dev/docs/>.
