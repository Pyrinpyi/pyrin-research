import "package:flutter/material.dart";
import "package:pyrin_app/components/tokens_list.dart";

import "../core/token.dart";
import "../ui.dart";


class SelectToken extends StatelessWidget
{
    final void Function(Token token) onClick;

    SelectToken({Key? key, required this.onClick});

  @override
    Widget build(BuildContext context)
    {
        return Container(
          width: double.infinity,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text("Select Token", style: Theme.of(context).textTheme.bodyMedium!.copyWith(fontWeight: FontWeight.w600)),
              const SizedBox(height: 20),
              PyrinTextField(
                name: "Search",
                hintText: "Search token",
                iconButton: PyrinIconButton(
                  icon: "search",
                ),
              ),
              const SizedBox(height: 20),
              TokensList(onClick: onClick),
            ],
          ),
        );
    }
}
