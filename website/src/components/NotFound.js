import React from "react";
import { useStaticQuery, graphql } from "gatsby";
import { Container, Row, Col, Button } from "react-bootstrap";
import Img from "gatsby-image";
import "../scss/_hero.scss";

export default function NotFound() {
  const data = useStaticQuery(graphql`
    {
      file(relativePath: { eq: "404.png" }) {
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
    <Container fluid className="notFound">
      <Container>
        <Row>
          <Col xs={{ span: 12, order: 2 }} md={{ span: 6, order: 1 }}>
            <h1>
              Hmmmm.....
              <br />
              Something's Not Right
            </h1>
            <p>
              Not sure how you got here. However, here is not here. Perhaps
              there is where you want to be. Where is there? There is an idea.
            </p>
            <Button style={{ width: 240 }}>Please Take Me Home</Button>
          </Col>
          <Col xs={{ span: 12, order: 1 }} md={{ span: 6, order: 2 }}>
            <Img fluid={data.file.childImageSharp.fluid} alt="404" />
          </Col>
        </Row>
      </Container>
    </Container>
  );
}
