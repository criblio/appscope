# AppScope Release Process

Our process is pretty straightforward. It focuses primarily on developing
the next release while keeping the effort devoted to maintaining older releases
to a minimum.

## Branching & Tagging

![Branching](images/branching.png)

* The default branch is `master`.

* We create branches from the default branch for defects `bug/123-name` and
  feature requests `feature/123-name`. The number prefix is an issue number
  and the `-name` usually aligns with the issue title. Occasionally, we skip
  creating an issue and omit the number.

* We create PRs when work on issue branches is complete and ready to be
  reviewed and merged back into the default branch.

* We name releases using the typical `major.minor.maintenance` pattern for
  [Semantic Versioning](https://semver.org/). We start with `0.0.0` and
  increment...

  * ... `major` when breaking changes (BCs) are made.
  * ... `minor` when we add features without BCs.
  * ... `maintenance` for bug fixes without BCs.

  We append `-rc#` suffixes for release candidates.

* We create release candidate tags `v*.*.0-rc*` on the default branch as we
  approach the next major or minor release. When it's ready to go, we create
  the release tag `v*.*.0`, also on the default branch.

* We create a release branch `release/0.x`, with the major number and `x`
  instead of the minor number. These are created from the default branch at the
  point where the corresponding `v*.0.0` tag was created. This provides a 
  starting point for future maintenance releases.

* Work toward a maintenance release is done on hotfix issue branches
  `hotfix/1234-name` created from the release branch, not the default branch.
  Like merges into the default branch, PRs are created to merge hotfixes back
  into the release branch when they're ready. Tags for the maintenance release
  candidates `v*.*.1-rc1` and the release itself `v*.*.1` are created on the
  release branch. Periodically, the release branch is merged back into the default branch.

* Minor releases after the `*.0.0` release are tagged on the default branch
  as usual — but after they're released, we merge them back out to the
  existing release branch. This blocks further maintenance releases for the
  prior minor release and provides a starting point for maintenance releases
  to the new minor release.

* For website content, the staging website always reflects what is on the
  master branch. The documentation team provides PRs that target the master
  branch. When they are merged, the website workflow runs, causing the
  staging website content to be updated.

## Workflows

We use GitHub Workflows and Actions to automate CI/CD tasks for the project
when changes are made in the repository.

The [`build`](../.github/workflows/build.yml) workflow builds the code and runs
the unit tests with every push, on every branch. When the tests pass, some
additional steps are taken depending on the trigger.

* The `scope` binary and other artifacts of the build are pushed to our
  [CDN](#cdn) for release tags and pushes to any branch.

* We build [container images](#container-images) and push them to Docker Hub
  for release tags.

* We run our [integration tests](../test/integration/) for pull requests to
  the default and release branches. We build and push the container images
  these tests use up to Docker Hub on pushes to the default branch.

The [`website`](../.github/workflows/website.yml) workflow handles building and
deploying the [`website/`](../website/) content to <https://staging.appscope.dev/>
and <https://appscope.dev/>. The staging website is intended to always reflect
the master branch. The production website is updated only when a "web" tag
has been applied and pushed. See the build script in that folder for details.

The [`update_latest`](../.github/workflows/update_latest.yml) workflow updates
the value returned by `https://cdn.cribl.io/dl/scope/latest`, and updates
the `latest` tag at `https://hub.docker.com/r/cribl/scope/tags`. This workflow
is run manually, and does not have any automatic triggers.

The [`update_latest_release`](../.github/workflows/update_latest_release.yml) workflow updates
the value returned by `https://cdn.cribl.io/dl/scope/latest-release`. This workflow
is run manually, and does not have any automatic triggers.

## CDN

You can [download](https://appscope.dev/docs/downloading#download-as-binary) the AppScope binary from the Cribl downloads page and follow the instructions there.

Or, you can use these CLI commands to directly download the binary and make it executable:

```text
LATEST=$(curl -Ls https://cdn.cribl.io/dl/scope/latest)
curl -Lo scope https://cdn.cribl.io/dl/scope/$LATEST/linux/$(uname -m)/scope
curl -Ls https://cdn.cribl.io/dl/scope/$LATEST/linux/$(uname -m)/scope.md5 | md5sum -c 
chmod +x scope
```

## Container Images

We build and push container images to the
[`cribl/scope`](https://hub.docker.com/r/cribl/scope) and
[`cribl/scope-demo`](https://hub.docker.com/r/cribl/scope-demo)
repositories at Docker Hub. See [`docker/`](../docker/) for details on how
those images are built.

We currently build these for release `v*` tags and tag the images to match with
the leading `v` stripped off.

```text
docker run --rm -it cribl/scope:latest
```
or
```text
docker run --rm -it cribl/scope:1.1.3
```

## Tag Usage

`next`
- Users can pick up the latest master build
- Created automatically from master
- Available on docker

`latest`
- Users can pick up the latest release after announcement
- Created manually on Release Day
- Available on docker and the CDN

`latest-release`
- Cribl Edge/Stream or Users can pick up the latest release before announcement
- Created manually on GA Ready Day
- Available on the CDN
 
