import {
  GithubOutlined,
  FileTextOutlined,
  FundProjectionScreenOutlined,
} from "@ant-design/icons";
import { Layout, Space, Tooltip } from "antd";

const { Header } = Layout;

function Top(props) {
  const { top_bg_color } = props;

  return (
    <Header className={"header"} style={{ backgroundColor: top_bg_color }}>
      <h1>
        XQAI-Eyes: Quantum Neural Network is no longer in the textbook!
        <br /> Explore how the Encoder works here.
      </h1>
      <Space
        size="large"
        style={{ display: "flex", justifyContent: "center", paddingBottom: 24 }}
      >
        <Tooltip title="GitHub">
          <a
            href="https://github.com/shaolunruan/XQAI-Eyes"
            target="_blank"
            rel="noopener noreferrer"
          >
            <GithubOutlined
              style={{ fontSize: 24, color: "rgba(255,255,255,0.7)" }}
            />
          </a>
        </Tooltip>
        <Tooltip title="Paper">
          <a
            href="https://arxiv.org/abs/2512.14181"
            target="_blank"
            rel="noopener noreferrer"
          >
            <FileTextOutlined
              style={{ fontSize: 24, color: "rgba(255,255,255,0.7)" }}
            />
          </a>
        </Tooltip>
        <Tooltip title="Slides presented in PacificVis 2026">
          <a
            href="https://docs.google.com/presentation/d/1zQ_5IcW6n-U_qwhsXX1tZf12Lvvqe6G2Hld8stsjEFQ/edit?usp=sharing"
            target="_blank"
            rel="noopener noreferrer"
          >
            <FundProjectionScreenOutlined
              style={{ fontSize: 24, color: "rgba(255,255,255,0.7)" }}
            />
          </a>
        </Tooltip>
      </Space>
    </Header>
  );
}

export default Top;
