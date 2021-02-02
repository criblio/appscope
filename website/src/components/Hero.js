import React from "react";
import { useStaticQuery, graphql, navigate } from "gatsby";
import { Container, Row, Col, Button } from "react-bootstrap";
import Img from "gatsby-image";
import "../scss/_hero.scss";

export default function Hero() {
  const data = useStaticQuery(graphql`
    {
      heroYaml {
        hero {
          ctaText
          header
          subText
        }
      }
      file(relativePath: { eq: "isoarchitecture.png" }) {
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
    <Container fluid className="hero">
      <Container>
        <Row>
          <Col xs={{ span: 12, order: 2 }} md={{ span: 6, order: 1 }}>
            <h1>{data.heroYaml.hero.header}</h1>
            <p>{data.heroYaml.hero.subText}</p>
            <Button
              onClick={() => {
                navigate("/docs/quick-start-guide");
              }}
            >
              {data.heroYaml.hero.ctaText}
            </Button>
          </Col>
          <Col xs={{ span: 12, order: 1 }} md={{ span: 6, order: 2 }}>
            <Img
              fluid={data.file.childImageSharp.fluid}
              alt="AppScope Processing Machine"
            />
          </Col>
        </Row>
      </Container>
    </Container>
  );
}
