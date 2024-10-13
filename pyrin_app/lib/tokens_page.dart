import "package:flutter/material.dart";
import "package:pyrin_app/components/tokens_list.dart";

import "../core/token.dart";
import "../ui.dart";
import "core/page.dart";


class TokensPage extends StatelessWidget
{
    @override
    Widget build(BuildContext context)
    {
      return RoutePage(
        name: "Tokens",
        child: Container(
          width: double.infinity,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              PyrinTextField(
                name: "Search",
                hintText: "Search token",
                iconButton: PyrinIconButton(
                  icon: "search",
                ),
              ),
              const SizedBox(height: 20),
              TokensList(),
            ],
          ),
        ),
      );
    }
}
