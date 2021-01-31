import React from "react";
import { useStaticQuery, graphql, navigate } from "gatsby";
import Img from "gatsby-image";
import { Container, Row, Col, Button } from "react-bootstrap";

export default function HowItWorks() {
  const data = useStaticQuery(graphql`
    query howItWorksQuery {
      allHowItWorksYaml {
        edges {
          node {
            title
            items {
              item
            }
            ctaText
            ctaLink
          }
        }
      }

      file(relativePath: { eq: "AppScope-system-level-design.png" }) {
        childImageSharp {
          fluid {
            sizes
            src
            srcSet
            base64
            aspectRatio
          }
        }
      }
    }
  `);

  return (
    <Container className="howItWorks">
      <Row>
        <Col xs={12} md={6} className="text-left">
          <h2>{data.allHowItWorksYaml.edges[0].node.title}</h2>
          {data.allHowItWorksYaml.edges[0].node.items.map((bullet, i) => {
            return <p>{bullet.item}</p>;
          })}
          <Button
            style={{ width: 230 }}
            onClick={() => {
              navigate(data.allHowItWorksYaml.edges[0].node.ctaLink);
            }}
          >
            {data.allHowItWorksYaml.edges[0].node.ctaText}
          </Button>
        </Col>

        <Col xs={12} md={6}>
          <Img
            fluid={data.file.childImageSharp.fluid}
            alt="AppScope Machine"
            style={{ maxWidth: 90 + "%", margin: "10px auto" }}
          />
        </Col>
      </Row>
    </Container>
  );
}
