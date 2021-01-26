import React, { useState } from "react";
import { Container, Col, Nav, NavDropdown } from "react-bootstrap";
import { useStaticQuery, graphql, navigate } from "gatsby";
import logo from "../../images/logo-cribl-new.svg";
import "../../scss/_criblNav.scss";

export default function CriblSiteNav() {
  const [isHovered, setIsHovered] = useState(false);
  const [isClicked, setIsClicked] = useState(false);
  const data = useStaticQuery(graphql`
    query CriblSiteNavQuery {
      allCorpSiteNavYaml {
        edges {
          node {
            navigationLeft {
              url
              parent
              child {
                link
                url
              }
            }
            navigationRight {
              parent
              child {
                link
                url
              }
            }
          }
        }
      }
    }
  `);
  return (
    <Container fluid className="criblNav-wrapper">
      <Container className="criblNav">
        <Col xs={12} md={6}>
          <Nav>
            <Nav.Item className="cribl-brand">
              <Nav.Link>
                <img src={logo} alt="Cribl" width="96" className="cribl-logo" />
              </Nav.Link>
            </Nav.Item>
            {data.allCorpSiteNavYaml.edges[0].node.navigationLeft.map(
              (item, i) => {
                return item.child === null ? (
                  <Nav.Item>
                    <Nav.Link key={i} href={item.url}>
                      {item.parent}
                    </Nav.Link>
                  </Nav.Item>
                ) : (
                  <Nav.Item>
                    <NavDropdown
                      title={item.parent}
                      className="dropdown"
                      key={i}
                    >
                      {item.child.map((childItem, j) => {
                        return (
                          <NavDropdown.Item
                            key={j}
                            onMouseEnter={() => setIsHovered(true)}
                            onMouseLeave={() => setIsHovered(false)}
                            onToggle={() => setIsClicked(!isClicked)}
                            show={isClicked || isHovered}
                          >
                            <Nav.Link href={childItem.url}>
                              {childItem.link}
                            </Nav.Link>
                          </NavDropdown.Item>
                        );
                      })}
                    </NavDropdown>
                  </Nav.Item>
                );
              }
            )}
          </Nav>
        </Col>
        <Col xs={12} md={6}>
          <Nav className="justify-content-end">
            <Nav.Item>
              {data.allCorpSiteNavYaml.edges[1].node.navigationRight.map(
                (item, i) => {
                  return item.child === null ? (
                    <Nav.Link key={i} href={item.url}>
                      {item.parent}
                    </Nav.Link>
                  ) : (
                    <NavDropdown
                      title={item.parent}
                      className="dropdown"
                      key={i}
                    >
                      {item.child.map((childItem, j) => {
                        return (
                          <NavDropdown.Item
                            key={j}
                            onMouseEnter={() => setIsHovered(true)}
                            onMouseLeave={() => setIsHovered(false)}
                            onToggle={() => setIsClicked(!isClicked)}
                            show={isClicked || isHovered}
                          >
                            <Nav.Link href={childItem.url}>
                              {childItem.link}
                            </Nav.Link>
                          </NavDropdown.Item>
                        );
                      })}
                    </NavDropdown>
                  );
                }
              )}
            </Nav.Item>
            <Nav.Link className="sandbox-btn">
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => {
                  navigate("https://sandbox.cribl.io/");
                }}
              >
                Try Our Sandbox
              </button>
            </Nav.Link>
          </Nav>
        </Col>
      </Container>
    </Container>
  );
}
