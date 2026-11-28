"""Clay's hosted MCP server for prospecting and enrichment."""
from app.agents.mcp.mcp_server_decorator import mcp_server
from app.agents.mcp.models import (
    MCPAuthMode,
    MCPCategory,
    MCPServerTemplate,
    MCPTransport,
)


@mcp_server(
    MCPServerTemplate(
        type_id="clay",
        display_name="Clay",
        description="Find and enrich companies and contacts, and run Clay workspace functions.",
        icon="/icons/connectors/clay.png",
        transport=MCPTransport.STREAMABLE_HTTP,
        default_auth_mode=MCPAuthMode.OAUTH,
        supported_auth_modes=[MCPAuthMode.OAUTH],
        default_url="https://api.clay.com/v3/mcp",
        # Clay's live /.well-known/oauth-authorization-server advertises DCR and PKCE S256.
        authorization_url="https://app.clay.com/oauth/authorize",
        token_url="https://api.clay.com/oauth/token",
        default_scopes=[],
        supports_dcr=True,
        documentation_url="https://university.clay.com/docs/connect-to-clay-mcp",
        tags=["prospecting", "enrichment", "sales"],
        category=MCPCategory.SALES,
        capabilities=[
            "Enrich company and contact data",
            "List workspace enrichment functions",
        ],
    )
)
class ClayMCPServer:
    """Register the Clay catalog template."""
