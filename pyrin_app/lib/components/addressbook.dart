import "dart:math" as math;
import "package:flutter/material.dart";
import "package:flutter/widgets.dart";
import "package:flutter_svg/flutter_svg.dart";
import "package:pyrin_app/core/addressbook.dart";

import "../core/token.dart";
import "../ui.dart";

class AddressBookList extends StatelessWidget
{
    final Function(AddressBook item)? onClick;

    AddressBookList({this.onClick});

    @override
    Widget build(BuildContext context)
    {
        final List<AddressBook> addresses =
        [
            AddressBook("Satoshi Nakamoto", "pyrintest:qqgjen34j4uqvyece06675mzxkc6fnaxrfmaptl6lcsmv6un7uc6vvwnsscs2"),
            AddressBook("Elon Musk", "pyrintest:qr8cu9mk3gpyrnuvtt4a7rpc75cr8esz6t62xxny8xjj3j88jtcqufpn9f8l4"),
        ];

        return Column(
          children: [
            PyrinTextField(
              name: "Address Book",
              hintText: "Search recipient",
              iconButton: PyrinIconButton(
                icon: "search",
              ),
            ),
            const SizedBox(height: 20),
            PyrinListView<AddressBook>(
                items: addresses,
                itemBuilder: (context, item)
                {
                  return PyrinListViewItemBuilder(
                    title: item.name,
                    subtitle: AddressBook.shortenAddress(item.address),
                    icon: Avatar(),
                  );
                }
            ),
          ],
        );
    }
}
