
class AddressBook
{
  final String name;
  final String address;

  AddressBook(this.name, this.address);

  static String shortenAddress(String address)
  {
      return address.substring(0, 5 + 4 + 1 + 8) + "..." + address.substring(address.length - 4);
  }
}
