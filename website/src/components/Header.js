import React from "react";
import { Container, Col, Nav } from "react-bootstrap";
import { useStaticQuery, graphql } from "gatsby";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faChevronDown } from "@fortawesome/free-solid-svg-icons";
import StarCount from "./widgets/StarCount";
import Download from "./widgets/Download";
import CriblSiteNav from "./criblSiteNav";
import logo from "../images/logo-appscope.svg";
import "../scss/_appScopeNav.scss";

export default function Header() {
  const data = useStaticQuery(graphql`
    query SiteNavQuery {
      allHeaderYaml {
        edges {
          node {
            name
            path
          }
        }
      }
    }
  `);

  return (
    <nav>
      <CriblSiteNav />
      <Container
        fluid
        className="nav-container"
        id="appscopeNav"
        style={{ backgroundColor: "#fff !important" }}
      >
        <Col>
          <Nav>
            <Nav.Item className="branding">
              <Nav.Link href="/">
                <img
                  className="appscope-logo"
                  src={logo}
                  alt="AppScope"
                  width="100"
                />
              </Nav.Link>
            </Nav.Item>
            {data.allHeaderYaml.edges.map((item, i) => {
              return (
                <Nav.Item key={i}>
                  <Nav.Link href={item.node.path}>{item.node.name} </Nav.Link>
                </Nav.Item>
              );
            })}
          </Nav>
        </Col>
        <Col className="appscope-nav-widgets">
          <StarCount />
          <Download btnText="Download" />
        </Col>
      </Container>
    </nav>
  );
}
