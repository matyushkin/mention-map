import { useEffect, useRef } from "react";
import * as d3 from "d3";
import type { GraphData } from "../utils/types";

interface Props {
  data: GraphData;
}

interface SimNode extends d3.SimulationNodeDatum {
  id: string;
  name: string;
  mentionCount: number;
}

interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  context: string;
}

export function MentionGraph({ data }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !data.characters.length) return;

    const width = 800;
    const height = 600;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const nodes: SimNode[] = data.characters.map((c) => ({
      id: c.id,
      name: c.name,
      mentionCount: c.mention_count,
    }));

    const nodeById = new Map(nodes.map((n) => [n.name, n]));

    const links: SimLink[] = data.mentions
      .filter((m) => nodeById.has(m.source) && nodeById.has(m.target))
      .map((m) => ({
        source: nodeById.get(m.source)!,
        target: nodeById.get(m.target)!,
        context: m.context,
      }));

    const simulation = d3
      .forceSimulation(nodes)
      .force("link", d3.forceLink(links).distance(100))
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(width / 2, height / 2));

    const link = svg
      .append("g")
      .attr("stroke", "#999")
      .attr("stroke-opacity", 0.6)
      .selectAll("line")
      .data(links)
      .join("line")
      .attr("stroke-width", 2);

    const node = svg
      .append("g")
      .selectAll("g")
      .data(nodes)
      .join("g")
      .call(
        d3
          .drag<SVGGElement, SimNode>()
          .on("start", (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
      );

    node
      .append("circle")
      .attr("r", (d) => Math.max(8, Math.min(30, d.mentionCount * 3)))
      .attr("fill", "#4a90d9")
      .attr("stroke", "#fff")
      .attr("stroke-width", 2);

    node
      .append("text")
      .text((d) => d.name)
      .attr("x", 0)
      .attr("y", -15)
      .attr("text-anchor", "middle")
      .attr("font-size", 12)
      .attr("fill", "#333");

    simulation.on("tick", () => {
      link
        .attr("x1", (d) => (d.source as SimNode).x!)
        .attr("y1", (d) => (d.source as SimNode).y!)
        .attr("x2", (d) => (d.target as SimNode).x!)
        .attr("y2", (d) => (d.target as SimNode).y!);

      node.attr("transform", (d) => `translate(${d.x},${d.y})`);
    });

    return () => {
      simulation.stop();
    };
  }, [data]);

  return (
    <div>
      <h2>Граф упоминаний</h2>
      <svg
        ref={svgRef}
        width={800}
        height={600}
        style={{ border: "1px solid #ddd", borderRadius: 8 }}
      />
    </div>
  );
}
