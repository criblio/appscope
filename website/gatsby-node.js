const path = require(`path`);
const { createFilePath } = require(`gatsby-source-filesystem`);

exports.onCreateNode = ({ node, getNode, actions }) => {
  const { createNodeField } = actions;
  if (node.internal.type === `MarkdownRemark`) {
    const slug = createFilePath({
      node,
      getNode,
      basePath: `pages/docs`,
    });
    createNodeField({
      node,
      name: `slug`,
      value: slug,
    });
  }
};

exports.createPages = async function ({ actions, graphql }) {
  const { data } = await graphql(`
    query {
      allMarkdownRemark {
        edges {
          node {
            id
            fields {
              slug
            }
          }
        }
      }
    }
  `);
  data.allMarkdownRemark.edges.forEach((edge) => {
    const slug = edge.node.fields.slug;
    const id = edge.node.id;
    actions.createPage({
      path: `docs` + slug,
      component: require.resolve(`./src/components/MarkDownBlock.js`),
      context: { slug: slug, id: id },
    });
  });

  actions.createRedirect({
    fromPath: '/docs',
    toPath: '/docs/overview',
    // redirectInBrowser: true,
    isPermanent: true,
  });
};
