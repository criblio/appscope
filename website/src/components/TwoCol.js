import React from "react";
import { useStaticQuery, graphql } from "gatsby";
import { Container, Row, Col, Card, Button } from "react-bootstrap";
import Img from "gatsby-image";
import "../scss/_highlights.scss";
import "../utils/font-awesome";

export default function TwoCol() {
  const data = useStaticQuery(graphql`
    {
      file(relativePath: { eq: "hero-image.png" }) {
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
    <>
      <Container className="highlights">
        <h2>Two Column Layout With Image On Left</h2>
        <p>Any subtext or descriptive text can be added here.</p>
        <Row>
          <Col md={3}>
            <Img
              fluid={data.file.childImageSharp.fluid}
              alt="A corgi smiling happily"
              style={{ maxWidth: 90 + "%", margin: "10px auto" }}
            />
          </Col>
          <Col md={9}>
            <p>
              Sed nec massa ante. Pellentesque non tortor sit amet purus
              fermentum aliquam. Nunc vel lacus id justo cursus semper id sed
              orci. Ut quis lacinia tellus, a lobortis sapien. Fusce a lacus et
              est finibus consectetur. Aliquam ipsum nisi, congue ut facilisis
              non, hendrerit quis justo. Nullam venenatis libero et nibh
              venenatis tincidunt. Nam lobortis diam vitae elit ullamcorper, sit
              amet efficitur felis varius.
            </p>
          </Col>
        </Row>
      </Container>

      <Container className="highlights">
        <h2>Two Column Layout With Image On Right & A Little Bigger</h2>
        <p>Any subtext or descriptive text can be added here.</p>
        <Row>
          <Col md={7}>
            <p>
              Sed nec massa ante. Pellentesque non tortor sit amet purus
              fermentum aliquam. Nunc vel lacus id justo cursus semper id sed
              orci. Ut quis lacinia tellus, a lobortis sapien. Fusce a lacus et
              est finibus consectetur. Aliquam ipsum nisi, congue ut facilisis
              non, hendrerit quis justo. Nullam venenatis libero et nibh
              venenatis tincidunt. Nam lobortis diam vitae elit ullamcorper, sit
              amet efficitur felis varius.
            </p>
          </Col>
          <Col md={5}>
            <Img
              fluid={data.file.childImageSharp.fluid}
              alt="A corgi smiling happily"
              style={{ maxWidth: 90 + "%", margin: "10px auto" }}
            />
          </Col>
        </Row>
      </Container>

      <Container
        fluid
        className="highlights-fluid"
        style={{
          color: "#fff",
          background: "rgb(33, 37, 41)",
        }}
      >
        <h2>Three Column Layout With Text Only</h2>
        <p>Any subtext or descriptive text can be added here.</p>
        <Row>
          <Col md={4}>
            <h4>Another Header...Or, Maybe Not</h4>
            <p>
              Sed nec massa ante. Pellentesque non tortor sit amet purus
              fermentum aliquam. Nunc vel lacus id justo cursus semper id sed
              orci. Ut quis lacinia tellus, a lobortis sapien. Fusce a lacus et
              est finibus consectetur. Aliquam ipsum nisi, congue ut facilisis
              non, hendrerit quis justo. Nullam venenatis libero et nibh
              venenatis tincidunt. Nam lobortis diam vitae elit ullamcorper, sit
              amet efficitur felis varius.
            </p>
          </Col>
          <Col md={4}>
            <h4>Another Header...Or, Maybe Not</h4>
            <p>
              Sed nec massa ante. Pellentesque non tortor sit amet purus
              fermentum aliquam. Nunc vel lacus id justo cursus semper id sed
              orci. Ut quis lacinia tellus, a lobortis sapien. Fusce a lacus et
              est finibus consectetur. Aliquam ipsum nisi, congue ut facilisis
              non, hendrerit quis justo. Nullam venenatis libero et nibh
              venenatis tincidunt. Nam lobortis diam vitae elit ullamcorper, sit
              amet efficitur felis varius.
            </p>
          </Col>
          <Col md={4}>
            <h4>Another Header...Or, Maybe Not</h4>
            <p>
              Sed nec massa ante. Pellentesque non tortor sit amet purus
              fermentum aliquam. Nunc vel lacus id justo cursus semper id sed
              orci. Ut quis lacinia tellus, a lobortis sapien. Fusce a lacus et
              est finibus consectetur. Aliquam ipsum nisi, congue ut facilisis
              non, hendrerit quis justo. Nullam venenatis libero et nibh
              venenatis tincidunt. Nam lobortis diam vitae elit ullamcorper, sit
              amet efficitur felis varius.
            </p>
          </Col>
        </Row>
      </Container>

      <Container className="highlights-cards">
        <h2>Card Components. Very good for a short description.</h2>
        <p>Any subtext or descriptive text can be added here.</p>
        <Row>
          <Col md={4}>
            <Card>
              <Img
                fluid={data.file.childImageSharp.fluid}
                alt="A corgi smiling happily"
                style={{ maxWidth: 90 + "%", margin: "10px auto" }}
              />

              <Card.Body>
                <Card.Title>Card Title</Card.Title>
                <Card.Text>
                  Some quick example text to build on the card title and make up
                  the bulk of the card's content.
                </Card.Text>
                <Button variant="primary">CALL TO ACTION</Button>
              </Card.Body>
            </Card>
          </Col>
          <Col md={4}>
            <Card>
              <Img
                fluid={data.file.childImageSharp.fluid}
                alt="A corgi smiling happily"
                style={{ maxWidth: 90 + "%", margin: "10px auto" }}
              />

              <Card.Body>
                <Card.Title>Card Title</Card.Title>
                <Card.Text>
                  Some quick example text to build on the card title and make up
                  the bulk of the card's content.
                </Card.Text>
                <Button variant="primary">CALL TO ACTION</Button>
              </Card.Body>
            </Card>
          </Col>

          <Col md={4}>
            <Card>
              <Img
                fluid={data.file.childImageSharp.fluid}
                alt="A corgi smiling happily"
              />

              <Card.Body>
                <Card.Title>Card Title</Card.Title>
                <Card.Text>
                  Some quick example text to build on the card title and make up
                  the bulk of the card's content.
                </Card.Text>
                <Button variant="primary">CALL TO ACTION</Button>
              </Card.Body>
            </Card>
          </Col>
        </Row>
      </Container>
      <Container>
        <Row>
          <Col md={7} className="blogPost-Card"></Col>
          <Col md={{ span: 4, offset: 1 }} className="blogPost-Card"></Col>
        </Row>
      </Container>
    </>
  );
}
